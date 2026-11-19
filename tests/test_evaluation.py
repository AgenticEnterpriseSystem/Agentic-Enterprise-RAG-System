import hashlib
import json
from datetime import datetime, timezone

from pydantic import ValidationError

from src.contracts import (
    AggregateEvidence, Answer, Chunk, PassageEvidence, QueryResponse, ScopeBinding,
    SnapshotRef, Trace, Usage,
)
from src.evaluation.experiment_contracts import (
    CodeProvenance, ExperimentManifest, GenerationBudgets, ModelProvenance,
    RetrievalProvenance, RuntimeCaseExport, RuntimeFailure, ScoringInput,
    ScoringReference, runtime_case_digest, validate_paired_arms,
    validate_resume_compatibility, validate_runtime_exports,
)

import pytest
from math import log2
from src.evaluation.retrieval_metrics import evaluate_retrieval, ndcg_at_k, recall_at_k
from collections import defaultdict
from src.evaluation.splits import build_splits


# Retrieval metrics

def test_hand_calculated_ranking_and_deduplication():
    # Unique ranking is a, irrelevant, b. Both a and b are relevant.
    hits = ["a", "a", "x", "b"]
    assert recall_at_k(hits, ["a", "b"], 2) == 0.5
    assert recall_at_k(hits, ["a", "b"], 3) == 1
    assert ndcg_at_k(hits, ["a", "b"], 3) == pytest.approx(1.5 / (1 + 1 / log2(3)))
    assert ndcg_at_k(["b", "a"], ["a", "b"], 2) == 1
    assert recall_at_k(["a"], ["a", "a"], 1) == 1


def test_empty_predictions_and_unlabeled():
    assert recall_at_k([], ["a"], 10) == 0
    assert ndcg_at_k([], ["a"], 10) == 0
    assert recall_at_k(["a"], [], 10) is None
    assert ndcg_at_k([], [], 10) is None


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k([], [], k)


def test_aggregation_includes_errors_and_reports_exclusions():
    rows = [
        {
            "question_id": "1",
            "question_type": "basic",
            "retrieved_doc_ids": ["a"],
            "relevant_doc_ids": ["a"],
        },
        {
            "question_id": "2",
            "question_type": "basic",
            "retrieved_doc_ids": ["a"],
            "relevant_doc_ids": ["a"],
            "error": "timeout",
        },
        {
            "question_id": "3",
            "question_type": "unknown",
            "retrieved_doc_ids": [],
            "relevant_doc_ids": [],
        },
    ]
    report = evaluate_retrieval(rows, ks=(10,))
    assert report["overall"]["recall@10"] == {"mean": 0.5, "scored": 2, "excluded": 1}
    assert report["overall"]["errors"] == 1
    assert report["slices"]["unknown"]["recall@10"]["mean"] is None
    with pytest.raises(ValueError):
        evaluate_retrieval(rows + rows)


# Question splits

def questions():
    return [
        {
            "question_id": f"q{i:03}",
            "question_type": f"type{i % 3}",
            "question": f"Question {i}",
            "expected_doc_ids": [f"d{i // 2}"],
        }
        for i in range(60)
    ]


def test_split_reproducible_disjoint_complete_and_grouped():
    rows = questions()
    rows[4]["expected_doc_ids"] = ["d1", "d2"]  # Transitive overlap.
    result = build_splits(rows)
    assert result == build_splits(list(reversed(rows)))
    splits = result["splits"]
    all_ids = [q for ids in splits.values() for q in ids]
    assert len(all_ids) == len(set(all_ids)) == len(rows)
    assignments = {q: name for name, ids in splits.items() for q in ids}
    doc_splits = defaultdict(set)
    for q in rows:
        for doc in q["expected_doc_ids"]:
            doc_splits[doc].add(assignments[q["question_id"]])
    assert all(len(names) == 1 for names in doc_splits.values())
    assert set(result["golden_development_ids"]) <= set(splits["development"])
    assert all(splits.values())


def test_duplicate_text_is_grouped_without_reference_docs():
    rows = questions()
    rows[0].update(question="same question", expected_doc_ids=[])
    rows[20].update(question=" SAME  question ", expected_doc_ids=[])
    result = build_splits(rows)
    assert any({"q000", "q020"} <= set(ids) for ids in result["splits"].values())


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        build_splits([questions()[0], questions()[0]])


def test_committed_split_manifest_is_consistent():
    import json
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "evaluation"
    manifest = json.loads((base / "splits.json").read_text())
    parts = manifest["splits"]
    ids = [qid for values in parts.values() for qid in values]
    assert len(ids) == len(set(ids)) == 500
    assert [len(parts[k]) for k in ("development", "calibration", "final")] == [
        300,
        100,
        100,
    ]
    assert len(manifest["golden_development_ids"]) == 60
    assert set(manifest["golden_development_ids"]) <= set(parts["development"])


# Hybrid variants over shared searches

def test_hybrid_comparison_skips_unlabeled_scores():
    from src.evaluation.run_hybrid import compare, recall10

    rows = [
        {"relevant_doc_ids": ["a"], "retrieved_doc_ids": ["a"]},
        {"relevant_doc_ids": ["b"], "retrieved_doc_ids": []},
        {"relevant_doc_ids": [], "retrieved_doc_ids": []},
        {"relevant_doc_ids": ["d"], "retrieved_doc_ids": ["d"]},
    ]
    baseline = [
        {"relevant_doc_ids": ["a"], "retrieved_doc_ids": []},
        {"relevant_doc_ids": ["b"], "retrieved_doc_ids": ["b"]},
        {"relevant_doc_ids": ["c"], "retrieved_doc_ids": ["c"]},
        {"relevant_doc_ids": [], "retrieved_doc_ids": []},
    ]
    assert compare(rows, baseline) == (1, 1)
    with pytest.raises(ValueError, match="labeled question"):
        recall10([])


def test_reranker_identity_requires_a_resolved_revision(monkeypatch):
    from types import SimpleNamespace
    from huggingface_hub import HfApi
    from src.evaluation.run_hybrid import reranker_identity

    monkeypatch.setattr(HfApi, 'model_info', lambda *a, **kw: SimpleNamespace(sha=None))
    with pytest.raises(ValueError, match="No revision"):
        reranker_identity('bge-reranker-base')
    revision = 'a' * 40
    monkeypatch.setattr(HfApi, 'model_info', lambda *a, **kw: SimpleNamespace(sha=revision))
    assert reranker_identity('bge-reranker-base')[2] == revision
    assert reranker_identity('bge-reranker-base@' + revision)[2] == revision


def test_hybrid_variants_share_searches_and_match_live_hybrid(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.contracts import SearchRequest, UserContext
    from src.evaluation.run_hybrid import run_single, run_variant, search_once
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.corpus_dense import CorpusDense, embed_corpus
    from src.retrieval.hybrid import HybridConfig, HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from tests.test_retrieval import FakeModel

    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table({
        "doc_id": [f"d{i}" for i in range(5)],
        "source_type": ["slack", "gmail", "linear", "slack", "jira"],
        "title": ["Deploy", "", "Sales", "Grid", "Canary"],
        "content": ["canary rollback steps for the deployment", "lunch menu canary rollback",
                    "quarterly forecast review", "elevation tokens for grid density",
                    "canary release checklist and rollback owner"],
        "path": ["p"] * 5,
    }), path)
    CorpusBM25.build(path).save(tmp_path / "bm25")
    bm25 = CorpusBM25.load(tmp_path / "bm25", corpus_path=path)
    embed_corpus(path, tmp_path / "dense", FakeModel(), shard_docs=2, window=4, overlap=1)
    dense = CorpusDense.load(tmp_path / "dense", FakeModel(), corpus_path=path, corpus=bm25.corpus)

    class Counting:
        def __init__(self, retriever):
            self.retriever, self.calls = retriever, 0
            self.doc_ids, self.manifest = retriever.doc_ids, {}

        def search(self, request):
            self.calls += 1
            return self.retriever.search(request)

    questions = [{"question_id": f"q{i}", "question_type": "basic", "source_types": ["slack"],
                  "question": text, "expected_doc_ids": ["d0"]}
                 for i, text in enumerate(["canary rollback", "grid density", "canary rollback"])]
    user = UserContext(user_id="benchmark", allowed_doc_ids=frozenset(bm25.doc_ids))
    counted = (Counting(bm25), Counting(dense))
    caches, seconds, errors = search_once(counted, questions, user, depth=5)
    assert [r.calls for r in counted] == [2, 2] and not errors  # Repeated question searched once.

    live = HybridRetriever(bm25, dense, config=HybridConfig(candidate_k=5, rerank_k=5))
    rows, latencies, manifest = run_variant(counted, caches, seconds, errors, questions, user,
                                            bm25_weight=0.5, top_k=5, candidate_k=5)
    assert manifest["method"] == "hybrid-rrf" and len(latencies) == 3
    for q, row in zip(questions, rows):
        expected = live.search(SearchRequest(query=q["question"], user=user, top_k=5))
        assert row["retrieved_doc_ids"] == [e.chunk.doc_id for e in expected]
    assert [r.calls for r in counted] == [2, 2]  # Variants replay the saved results.

    keyword_only, _, _ = run_variant(counted, caches, seconds, errors, questions, user,
                                     bm25_weight=1.0, top_k=5, candidate_k=5)
    bm25_rows, _ = run_single(caches[0], seconds, errors, questions, 5)
    assert [r["retrieved_doc_ids"] for r in keyword_only] == [r["retrieved_doc_ids"] for r in bm25_rows]

    # The reranker reorders the fused candidates; here it prefers the longest passage.
    reranker = CrossEncoderReranker("offline-test", "a" * 40,
                                    scorer=lambda pairs: [len(text) for _, text in pairs])
    reranked, _, manifest = run_variant(counted, caches, seconds, errors, questions[:1], user,
                                        bm25_weight=0.5, reranker=reranker, top_k=5, candidate_k=5)
    assert set(reranked[0]["retrieved_doc_ids"]) == set(rows[0]["retrieved_doc_ids"])
    assert reranked[0]["scores"] == sorted(reranked[0]["scores"], reverse=True)
    assert reranked[0]["scores"] != rows[0]["scores"]
    assert manifest["reranker"]["name"] == "offline-test"


# Screening sample

def test_sample_keeps_references_lookalikes_and_seeded_random_documents():
    from src.data.sample import sample_ids
    from src.evaluation.screen_models import misses_found

    corpus = [f"d{i}" for i in range(20)]
    questions = [{"question_id": "q1", "expected_doc_ids": ["d0"]},
                 {"question_id": "q2", "expected_doc_ids": ["d1", "d2"]}]
    rankings = [[{"question_id": "q1", "retrieved_doc_ids": ["d3", "d0", "d4"]},
                 {"question_id": "other", "retrieved_doc_ids": ["d19"]}],  # Not in the split.
                [{"question_id": "q2", "retrieved_doc_ids": ["d5", "d6", "d7"]}]]
    chosen, counts = sample_ids(corpus, questions, rankings, top_k=2, random_docs=3, seed=1)
    # References plus each question's top 2 (d4 and d7 rank third); "other" is not in the split.
    assert {"d0", "d1", "d2", "d3", "d5", "d6"} <= chosen
    assert counts == {"reference": 3, "lookalikes": 3, "random": 3} and len(chosen) == 9
    again, _ = sample_ids(corpus, questions, rankings, top_k=2, random_docs=3, seed=1)
    assert again == chosen  # Seeded, so the sample is reproducible.
    with pytest.raises(ValueError, match="missing"):
        sample_ids(corpus, [{"question_id": "q", "expected_doc_ids": ["nope"]}], [], random_docs=1)

    bm25 = [{"relevant_doc_ids": ["a"], "retrieved_doc_ids": ["x"]},
            {"relevant_doc_ids": ["b"], "retrieved_doc_ids": ["b"]},
            {"relevant_doc_ids": [], "retrieved_doc_ids": []}]
    dense = [{"relevant_doc_ids": ["a"], "retrieved_doc_ids": ["a"]},
             {"relevant_doc_ids": ["b"], "retrieved_doc_ids": ["y"]},
             {"relevant_doc_ids": [], "retrieved_doc_ids": []}]
    assert misses_found(dense, bm25) == (1, 1)


# Immutable experiment exports and external scoring handoff

EXPERIMENT_NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)
EXPERIMENT_SHA = "a" * 64
EXPERIMENT_OTHER_SHA = "b" * 64
EXPERIMENT_COMMIT = "c" * 40
EXPERIMENT_SNAPSHOT = SnapshotRef(snapshot_id="corpus-v1", corpus_sha256=EXPERIMENT_SHA,
                       catalog_sha256=EXPERIMENT_OTHER_SHA, document_count=2)
EXPERIMENT_SCOPE = ScopeBinding(scope_id="benchmark", user_id="reviewer", snapshot=EXPERIMENT_SNAPSHOT,
                     policy_revision=1, session_version=1)


def _contract_update(model, **changes):
    return type(model).model_validate({**model.model_dump(), **changes})


def _experiment_manifest(workflow="fixed", **changes):
    values = dict(experiment_id="comparison", manifest_id=workflow, workflow=workflow,
        created_at=EXPERIMENT_NOW, code=CodeProvenance(commit=EXPERIMENT_COMMIT), split="development",
        split_manifest_sha256=EXPERIMENT_SHA, question_set_sha256=EXPERIMENT_SHA, question_ids=("q1", "q2"),
        scope=EXPERIMENT_SCOPE, scope_fingerprint_sha256=EXPERIMENT_SHA,
        generation_model=ModelProvenance(provider="scripted", model_id="model-v1"),
        generator_prompt_sha256=EXPERIMENT_SHA, critic_prompt_sha256=EXPERIMENT_SHA if workflow == "adaptive" else None,
        comparison_config_sha256=EXPERIMENT_SHA,
        retrieval=RetrievalProvenance(retriever_id="hybrid", snapshot=EXPERIMENT_SNAPSHOT,
            index_manifest_sha256=EXPERIMENT_SHA, config_sha256=EXPERIMENT_SHA),
        budgets=GenerationBudgets(top_k=5, max_question_chars=8000,
            max_context_chars=24000, max_output_tokens=1024, max_run_tokens=16000,
            max_run_seconds=120, max_run_cost_usd=.1), max_rounds=3 if workflow == "adaptive" else 1)
    return ExperimentManifest(**{**values, **changes})


def _experiment_passage(snapshot_id=EXPERIMENT_SNAPSHOT.snapshot_id):
    return PassageEvidence(citation_id="doc:0:8", snapshot_id=snapshot_id, revision_id="revision-v1",
        chunk=Chunk(chunk_id="doc:0:8", doc_id="doc", source_type="jira", title="Ticket",
                    text="Resolved", start_char=0, end_char=8), score=1, retrieval_method="hybrid")


def _runtime_case(owner="fixed", qid="q1", *, failed=False, evidence=None, **changes):
    selected = (_experiment_passage(),) if evidence is None else evidence
    question = "What is the status?" if qid == "q1" else "What happened?"
    run_id = f"{owner}-{qid}"
    response = QueryResponse(snapshot=EXPERIMENT_SNAPSHOT,
        answer=Answer(status="error" if failed else "answered",
            text="Model unavailable" if failed else "Resolved.",
            citation_ids=() if failed else (selected[0].citation_id,)),
        sources=() if failed else selected,
        trace=Trace(run_id=run_id, stop_reason="error" if failed else "single_pass",
                    usage=Usage(api_calls=1, estimated_cost_usd=None)))
    values = dict(manifest_id=owner, run_id=run_id, question_id=qid, question=question,
        question_sha256=hashlib.sha256(question.encode()).hexdigest(), response=response,
        generation_evidence=() if failed else selected,
        error=RuntimeFailure(code="provider_unavailable", stage="generate") if failed else None)
    return RuntimeCaseExport(**{**values, **changes})


def test_export_round_trip_keeps_failures_unknown_costs_and_exact_generation_evidence():
    fixed = _experiment_manifest()
    cases = (_runtime_case(), _runtime_case(qid="q2", failed=True))
    validate_runtime_exports(fixed, cases, require_complete=True)
    restored = RuntimeCaseExport.model_validate_json(cases[1].model_dump_json())
    assert restored == cases[1]
    assert restored.response.trace.usage.estimated_cost_usd is None
    assert restored.error.code == "provider_unavailable"
    assert cases[0].generation_evidence == cases[0].response.sources
    assert ExperimentManifest.model_validate_json(fixed.model_dump_json()) == fixed


def test_partial_resume_is_distinct_from_a_missing_complete_outcome():
    validate_runtime_exports(_experiment_manifest(), ())
    validate_runtime_exports(_experiment_manifest(), (_runtime_case(),))
    with pytest.raises(ValueError, match="missing declared"):
        validate_runtime_exports(_experiment_manifest(), (_runtime_case(),), require_complete=True)
    validate_resume_compatibility(_experiment_manifest(), _experiment_manifest(created_at=datetime(2026, 11, 3, tzinfo=timezone.utc)))


@pytest.mark.parametrize("field,value", [
    ("manifest_id", "another-run"), ("split", "final"), ("question_ids", ("q2", "q1")),
    ("scope_fingerprint_sha256", EXPERIMENT_OTHER_SHA), ("generator_prompt_sha256", EXPERIMENT_OTHER_SHA),
    ("code", CodeProvenance(commit="d" * 40)),
])
def test_resume_rejects_changed_identity_or_scientific_settings(field, value):
    with pytest.raises(ValueError, match="Resume changes"):
        validate_resume_compatibility(_experiment_manifest(), _experiment_manifest(**{field: value}))


def test_frozen_split_and_full_source_commit_are_required():
    for split in ("development", "calibration", "final"):
        assert _experiment_manifest(split=split).split == split
    for split in ("test", "train", "FINAL"):
        with pytest.raises(ValidationError):
            _experiment_manifest(split=split)
    for commit in ("da970e8", "g" * 40, "c" * 39, "C" * 40):
        with pytest.raises(ValidationError):
            CodeProvenance(commit=commit)
    with pytest.raises(ValidationError, match="diff hash"):
        CodeProvenance(commit=EXPERIMENT_COMMIT, dirty=True)
    with pytest.raises(ValidationError, match="diff hash"):
        CodeProvenance(commit=EXPERIMENT_COMMIT, diff_sha256=EXPERIMENT_SHA)
    assert CodeProvenance(commit=EXPERIMENT_COMMIT, dirty=True, diff_sha256=EXPERIMENT_SHA).dirty


def test_selection_snapshot_and_arm_metadata_cannot_contradict_the_manifest():
    with pytest.raises(ValidationError, match="Duplicate manifest"):
        _experiment_manifest(question_ids=("q1", "q1"))
    other = _contract_update(EXPERIMENT_SNAPSHOT, catalog_sha256="d" * 64)
    with pytest.raises(ValidationError, match="same snapshot"):
        _experiment_manifest(retrieval=_contract_update(_experiment_manifest().retrieval, snapshot=other))
    with pytest.raises(ValidationError, match="one round"):
        _experiment_manifest(max_rounds=2)
    with pytest.raises(ValidationError, match="critic prompt"):
        _experiment_manifest("adaptive", critic_prompt_sha256=None)


@pytest.mark.parametrize("extra", ["reference_answer", "gold_answer", "answer_facts", "reference"])
def test_reference_fields_are_forbidden_in_runtime_exports(extra):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeCaseExport.model_validate({**_runtime_case().model_dump(), extra: "Scoring only"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentManifest.model_validate({**_experiment_manifest().model_dump(), extra: "Scoring only"})


def test_nested_contracts_are_immutable_and_do_not_retain_mutable_input_lists():
    ids = ["q1", "q2"]
    frozen = _experiment_manifest(question_ids=ids)
    ids.append("q3")
    assert frozen.question_ids == ("q1", "q2")
    for obj, field, value in ((frozen, "split", "final"), (frozen.budgets, "top_k", 10),
        (frozen.scope.snapshot, "document_count", 999), (_runtime_case().response.answer, "text", "Changed")):
        with pytest.raises(ValidationError, match="frozen_instance"):
            setattr(obj, field, value)
    json.dumps(ExperimentManifest.model_json_schema())
    json.dumps(RuntimeCaseExport.model_json_schema())
    json.dumps(ScoringInput.model_json_schema())


def test_duplicate_question_and_run_ids_reject_while_error_outcomes_remain_valid():
    first = _runtime_case()
    with pytest.raises(ValueError, match="Duplicate question or run"):
        validate_runtime_exports(_experiment_manifest(), (first, first))
    second = _runtime_case(qid="q2")
    reused = _contract_update(second, run_id=first.run_id,
                    response=_contract_update(second.response, trace=_contract_update(second.response.trace, run_id=first.run_id)))
    with pytest.raises(ValueError, match="Duplicate question or run"):
        validate_runtime_exports(_experiment_manifest(), (first, reused))
    with pytest.raises(ValueError, match="another manifest"):
        validate_runtime_exports(_experiment_manifest(), (_runtime_case(owner="adaptive"),))
    with pytest.raises(ValueError, match="frozen selection"):
        validate_runtime_exports(_experiment_manifest(), (_runtime_case(qid="outside"),))


def test_invalid_runtime_metadata_is_not_confused_with_a_valid_failed_case():
    failed = _runtime_case(failed=True)
    validate_runtime_exports(_experiment_manifest(), (failed,))
    with pytest.raises(ValidationError, match="run ID"):
        _contract_update(failed, run_id="another-run")
    with pytest.raises(ValidationError, match="Question text"):
        _contract_update(failed, question_sha256=EXPERIMENT_OTHER_SHA)
    with pytest.raises(ValidationError, match="failure metadata"):
        _contract_update(failed, error=None)
    with pytest.raises(ValidationError, match="failure metadata"):
        _contract_update(_runtime_case(), error=failed.error)


def test_error_trace_and_error_answer_must_agree_in_both_directions():
    succeeded = _runtime_case()
    with pytest.raises(ValidationError, match="error trace"):
        _contract_update(succeeded, response=_contract_update(succeeded.response,
            trace=_contract_update(succeeded.response.trace, stop_reason="error")))
    failed = _runtime_case(failed=True)
    with pytest.raises(ValidationError, match="error trace"):
        _contract_update(failed, response=_contract_update(failed.response,
            trace=_contract_update(failed.response.trace, stop_reason="single_pass")))


def test_direct_clarification_preserves_citations_without_inventing_generation_input():
    original = _runtime_case()
    clarification = _contract_update(original.response,
        answer=Answer(status="insufficient_evidence", text="Which incident do you mean?",
                      citation_ids=original.response.answer.citation_ids),
        trace=_contract_update(original.response.trace, stop_reason="clarification"))
    exported = _contract_update(original, response=clarification, generation_evidence=())
    validate_runtime_exports(_experiment_manifest(), (exported,))
    assert exported.response.answer.citation_ids and not exported.generation_evidence
    with pytest.raises(ValidationError, match="no generation input"):
        _contract_update(exported, generation_evidence=original.generation_evidence)


def test_public_helpers_revalidate_copied_cases_instead_of_trusting_frozen_types():
    copied = _runtime_case().model_copy(update={"question": "Substituted text with the original hash"})
    with pytest.raises(ValidationError, match="Question text"):
        validate_runtime_exports(_experiment_manifest(), (copied,))
    with pytest.raises(ValidationError, match="Question text"):
        validate_paired_arms(_experiment_manifest(), _experiment_manifest("adaptive"), (copied,),
                             (_runtime_case(owner="adaptive"),), require_complete=False)
    with pytest.raises(ValidationError, match="Question text"):
        runtime_case_digest(copied)


def test_public_helpers_revalidate_copied_manifest_and_nested_metadata():
    bad_rounds = _experiment_manifest().model_copy(update={"max_rounds": 2})
    with pytest.raises(ValidationError, match="one round"):
        validate_runtime_exports(bad_rounds, ())
    with pytest.raises(ValidationError, match="one round"):
        validate_resume_compatibility(_experiment_manifest(), bad_rounds)
    with pytest.raises(ValidationError, match="one round"):
        validate_paired_arms(bad_rounds, _experiment_manifest("adaptive"), (), (), require_complete=False)
    nested = _experiment_manifest().model_copy(update={
        "code": CodeProvenance(commit=EXPERIMENT_COMMIT).model_copy(update={"commit": "short"})})
    with pytest.raises(ValidationError):
        validate_runtime_exports(nested, ())


def test_generation_evidence_must_match_answer_and_snapshot():
    with pytest.raises(ValidationError, match="outside the recorded"):
        _contract_update(_runtime_case(), generation_evidence=())
    with pytest.raises(ValidationError, match="Duplicate generation"):
        _contract_update(_runtime_case(), generation_evidence=(_experiment_passage(), _experiment_passage()))
    with pytest.raises(ValidationError, match="another snapshot"):
        _contract_update(_runtime_case(), generation_evidence=(_experiment_passage("other-snapshot"),))
    changed = _contract_update(_experiment_passage(), chunk=_contract_update(_experiment_passage().chunk, text="Deferred"))
    with pytest.raises(ValidationError, match="evidence disagree"):
        _contract_update(_runtime_case(), generation_evidence=(changed,))
    other = _contract_update(EXPERIMENT_SNAPSHOT, catalog_sha256="d" * 64)
    with pytest.raises(ValueError, match="another snapshot"):
        validate_runtime_exports(_experiment_manifest(), (_contract_update(_runtime_case(), response=_contract_update(_runtime_case().response, snapshot=other)),))


def test_zero_aggregate_keeps_query_scope_provenance_and_validates_as_runtime_evidence():
    zero = AggregateEvidence(citation_id="aggregate-zero", artifact_id="artifact-zero", scope=EXPERIMENT_SCOPE,
        template_id="ticket-count", template_version="1", count=0, lineage_sha256=EXPERIMENT_SHA,
        coverage="verified_records", created_at=EXPERIMENT_NOW)
    exported = _runtime_case(evidence=(zero,))
    validate_runtime_exports(_experiment_manifest(), (exported,))
    roundtrip = RuntimeCaseExport.model_validate_json(exported.model_dump_json())
    assert roundtrip.generation_evidence[0].count == 0
    assert roundtrip.generation_evidence[0].scope == EXPERIMENT_SCOPE
    other_user = _contract_update(zero, scope=_contract_update(EXPERIMENT_SCOPE, user_id="different-user"))
    with pytest.raises(ValueError, match="another permission scope"):
        validate_runtime_exports(_experiment_manifest(), (_runtime_case(evidence=(other_user,)),))


def test_pairs_allow_different_answers_order_and_retained_failure_but_not_unmatched_cases():
    left, right = _experiment_manifest(), _experiment_manifest("adaptive")
    fixed_cases = (_runtime_case(), _runtime_case(qid="q2", failed=True))
    adaptive_cases = (_runtime_case(owner="adaptive", qid="q2"), _runtime_case(owner="adaptive"))
    validate_paired_arms(left, right, fixed_cases, adaptive_cases)
    with pytest.raises(ValueError, match="missing declared"):
        validate_paired_arms(left, right, fixed_cases, adaptive_cases[:1])
    with pytest.raises(ValueError, match="unmatched question"):
        validate_paired_arms(left, right, fixed_cases[:1], adaptive_cases[:1], require_complete=False)


@pytest.mark.parametrize("field,value", [
    ("code", CodeProvenance(commit="d" * 40)),
    ("scope", _contract_update(EXPERIMENT_SCOPE, policy_revision=2)),
    ("generation_model", ModelProvenance(provider="scripted", model_id="different-model")),
    ("generator_prompt_sha256", EXPERIMENT_OTHER_SHA), ("comparison_config_sha256", EXPERIMENT_OTHER_SHA),
    ("router_prompt_sha256", EXPERIMENT_OTHER_SHA), ("temperature", .5), ("split", "calibration"),
    ("budgets", _contract_update(_experiment_manifest().budgets, max_context_chars=12000)),
    ("retrieval", _contract_update(_experiment_manifest().retrieval, config_sha256=EXPERIMENT_OTHER_SHA)),
    ("retrieval", _contract_update(_experiment_manifest().retrieval, index_manifest_sha256=EXPERIMENT_OTHER_SHA)),
    ("question_ids", ("q1",)),
])
def test_pairs_reject_incompatible_scope_model_retrieval_evidence_budgets_and_config(field, value):
    with pytest.raises(ValueError, match="shared configuration or question selection"):
        validate_paired_arms(_experiment_manifest(), _experiment_manifest("adaptive", **{field: value}), (), (), require_complete=False)


def test_pair_identity_and_question_text_cannot_be_reused_or_substituted():
    with pytest.raises(ValueError, match="distinct manifest"):
        validate_paired_arms(_experiment_manifest(), _experiment_manifest("adaptive", manifest_id="fixed"), (), ())
    with pytest.raises(ValueError, match="fixed and adaptive"):
        validate_paired_arms(_experiment_manifest("adaptive"), _experiment_manifest(), (), ())
    left, right = _runtime_case(), _runtime_case(owner="adaptive")
    right = _contract_update(right, run_id=left.run_id,
                   response=_contract_update(right.response, trace=_contract_update(right.response.trace, run_id=left.run_id)))
    with pytest.raises(ValueError, match="reused across paired"):
        validate_paired_arms(_experiment_manifest(), _experiment_manifest("adaptive"), (left,), (right,), require_complete=False)
    question = "A substituted question"
    right = _contract_update(_runtime_case(owner="adaptive"), question=question,
                   question_sha256=hashlib.sha256(question.encode()).hexdigest())
    with pytest.raises(ValueError, match="question text differs"):
        validate_paired_arms(_experiment_manifest(), _experiment_manifest("adaptive"), (left,), (right,), require_complete=False)


def test_scoring_inputs_are_a_separate_exact_hash_join_and_allow_absent_references():
    runtime = _runtime_case()
    reference = ScoringReference(question_id=runtime.question_id, question_sha256=runtime.question_sha256,
        reference_set_sha256=EXPERIMENT_SHA, reference_answer="The ticket was resolved.",
        answer_facts=("Resolved",), expected_doc_ids=("doc",))
    scoring = ScoringInput(runtime_case=runtime, runtime_case_sha256=runtime_case_digest(runtime), reference=reference)
    assert ScoringInput.model_validate_json(scoring.model_dump_json()) == scoring
    assert "reference_answer" not in runtime.model_dump_json()
    with pytest.raises(ValidationError, match="another question"):
        _contract_update(scoring, reference=_contract_update(reference, question_id="q2"))
    with pytest.raises(ValidationError, match="different runtime case"):
        _contract_update(scoring, runtime_case_sha256=EXPERIMENT_OTHER_SHA)
    absent = _contract_update(reference, reference_answer=None, answer_facts=(), expected_doc_ids=())
    assert _contract_update(scoring, reference=absent).reference.reference_answer is None

def test_runtime_export_requires_snapshot_even_when_legacy_responses_do_not():
    failed = _runtime_case(failed=True)
    legacy_response = QueryResponse(
        answer=Answer(status="error", text="Unavailable"),
        trace=Trace(run_id=failed.run_id, stop_reason="error",
                    usage=Usage(api_calls=0, estimated_cost_usd=None)),
    )
    assert legacy_response.snapshot is None
    with pytest.raises(ValidationError, match="snapshot-bound response"):
        _contract_update(failed, response=legacy_response)
    unvalidated = failed.model_copy(update={"response": legacy_response})
    with pytest.raises(ValidationError, match="snapshot-bound response"):
        validate_runtime_exports(_experiment_manifest(), (unvalidated,))
