"""Offline development comparison; never generates answers or rebuilds indexes."""
import argparse
import json
import resource
import re
import sys
import time
from pathlib import Path

from src.config import ROOT, runtime_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--memory", action="store_true", help="Full dense load in a separate process; no model")
    parser.add_argument("--legacy-ties", action="store_true", help="Replay the old cutoff rule to isolate passage changes")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("Label must contain only letters, digits, underscores, or hyphens")
    out = ROOT / ".cache/fix-comparison"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / (args.label + ("-memory" if args.memory else "") + ".json")
    if destination.exists():
        parser.error("Output exists; use a new label to preserve measurements")
    corpus = ROOT / ".cache/corpus"
    started = time.perf_counter()
    if args.memory:
        import numpy as np
        from src.retrieval.corpus_dense import CorpusDense
        dense = CorpusDense.load(corpus / "indexes/dense")
        load_seconds = time.perf_counter() - started
        vector = np.load(corpus / "indexes/dense/shard_00000.npy", mmap_mode="r")[0].astype(np.float32)
        vector /= np.linalg.norm(vector)
        result = {"mode": "dense-load-no-model", "load_seconds": load_seconds,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                  "passages": dense.index.ntotal, "sample_rank": dense.rank_vector(vector, 20)}
    else:
        from src.evaluation.splits import select_questions
        from src.contracts import GenerationRequest, SearchRequest, UserContext
        from src.evaluation.run_retrieval import load_questions
        from src.generation.generator import select_context
        from src.retrieval.corpus_bm25 import CorpusBM25
        questions = load_questions(corpus / "questions.jsonl", ROOT / "evaluation/splits.json", "development")
        selected = select_questions(questions, 20)
        sample_path = out / "questions.json"
        ids = [q["question_id"] for q in selected]
        if sample_path.exists():
            assert json.loads(sample_path.read_text()) == ids, "Question sample changed"
        else:
            sample_path.write_text(json.dumps(ids, indent=2) + "\n")
        retriever = CorpusBM25.load(corpus / "indexes/bm25", corpus_path=corpus / "documents.parquet")
        if args.legacy_ties:
            import numpy as np
            from src.retrieval.corpus_bm25 import tokenize

            def old_rank(query, top_k, mask=None):
                ids = [retriever.model.vocab_dict[t] for t in tokenize(query) if t in retriever.model.vocab_dict]
                if not ids:
                    return []
                offset = float(retriever.model.nonoccurrence_array[ids].sum())
                matched = retriever.model.get_scores_from_ids(ids) - offset
                if mask is not None:
                    matched[~mask] = 0
                candidates = np.flatnonzero(matched > 0)
                if len(candidates) > top_k:
                    candidates = candidates[np.argpartition(-matched[candidates], top_k - 1)[:top_k]]
                return [(retriever.doc_ids[i], float(matched[i]) + offset)
                        for i in sorted(candidates, key=lambda i: (-matched[i], retriever.doc_ids[i]))]
            retriever.rank = old_rank
        user = UserContext(user_id="offline-comparison", allowed_doc_ids=frozenset(retriever.doc_ids))
        rows = []
        for q in selected:
            began = time.perf_counter()
            hits = retriever.search(SearchRequest(query=q["question"], user=user, top_k=20))
            latency = time.perf_counter() - began
            context = select_context(GenerationRequest(question=q["question"], user=user, evidence=hits[:5]), 24000)
            for item in context:
                original = retriever._chunk(item.chunk.doc_id)
                assert original.chunk_id  # Original exact title/content representation.
                assert item.chunk.text == original.text[item.chunk.start_char:item.chunk.end_char]
            gold = set(q["expected_doc_ids"])
            rows.append({"question_id": q["question_id"], "question_type": q["question_type"],
                         "retrieved_ids": [e.chunk.doc_id for e in hits], "search_seconds": latency,
                         "retrieved_gold": sorted(gold & {e.chunk.doc_id for e in hits}),
                         "context_gold": sorted(gold & {e.chunk.doc_id for e in context}),
                         "context_chars": sum(len(e.chunk.text) for e in context),
                         "selected": [{"doc_id": e.chunk.doc_id, "citation_id": e.citation_id,
                                       "start_char": e.chunk.start_char, "end_char": e.chunk.end_char,
                                       "text": e.chunk.text} for e in context]})
        result = {"mode": "real-bm25-search", "settings": {"top_k": 20, "context_top_k": 5, "context_chars": 24000,
                                                          "legacy_ties": args.legacy_ties},
                  "corpus": retriever.manifest["corpus_sha256"], "rows": rows,
                  "limitations": ["Document relevance does not establish passage support or answer quality.",
                                   "No LLM or cross-encoder calls; no dense query model needed."]}
    result["versions"] = runtime_versions()["packages"]
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)
    if args.memory and (out / "before-memory.json").exists() and args.label != "before":
        before = json.loads((out / "before-memory.json").read_text())
        comparison = {"before_peak_rss_bytes": before["peak_rss_bytes"],
                      "after_peak_rss_bytes": result["peak_rss_bytes"],
                      "before_load_seconds": before["load_seconds"], "after_load_seconds": result["load_seconds"],
                      "sample_rank_unchanged": before["sample_rank"] == result["sample_rank"],
                      "limitations": "Separate single runs; cache and host load can affect time and RSS."}
        (out / (args.label + "-memory-comparison.json")).write_text(json.dumps(comparison, indent=2) + "\n")
    if not args.memory and (out / "before.json").exists() and args.label != "before":
        before = json.loads((out / "before.json").read_text())
        comparison = {"label": args.label, "questions": len(result["rows"]),
                      "changed_rankings": sum(a["retrieved_ids"] != b["retrieved_ids"] for a, b in zip(before["rows"], result["rows"])),
                      "before_context_gold_documents": sum(len(r["context_gold"]) for r in before["rows"]),
                      "after_context_gold_documents": sum(len(r["context_gold"]) for r in result["rows"]),
                      "before_context_chars": sum(r["context_chars"] for r in before["rows"]),
                      "after_context_chars": sum(r["context_chars"] for r in result["rows"]),
                      "before_search_seconds": sum(r["search_seconds"] for r in before["rows"]),
                      "after_search_seconds": sum(r["search_seconds"] for r in result["rows"])}
        (out / (args.label + "-comparison.json")).write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
