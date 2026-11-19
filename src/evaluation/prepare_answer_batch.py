"""Prepare legacy BM25 development evidence without overwriting saved results.

This preserves Alex's passage-only preparation format. Snapshot-bound runtime
exports use experiment_io; this helper does not invent missing provenance.
"""

import argparse
import json
import os
from pathlib import Path

from src.config import ROOT, CORPUS_DIR


def _source_path(source_root: Path, relative: str) -> Path:
    path = (source_root / relative.replace('\\', '/')).resolve()
    try:
        path.relative_to(source_root.resolve())
    except ValueError:
        raise ValueError('Document index escapes the benchmark source directory') from None
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        name = str(path)
        path = Path('\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\') else '\\\\?\\' + name)
    return path


def prepare_cases(selected_path: Path, predictions_path: Path, benchmark: Path, *,
                  top_k: int = 5, expected_count: int | None = 20) -> list[dict]:
    # Keep CLI --help independent of corpus/model/index imports.
    from src.data.corpus import extract_document

    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k must be an integer between 1 and 100')
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ValueError('Expected question count must be positive')
    selected = json.loads(Path(selected_path).read_text(encoding='utf-8'))
    if not isinstance(selected, list) or not selected:
        raise ValueError('Selected questions must be a nonempty JSON list')
    ids = [row.get('question_id') for row in selected if isinstance(row, dict)]
    if (len(ids) != len(selected) or any(not isinstance(qid, str) or not qid.strip() for qid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Selected questions need distinct nonblank IDs')
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError('Selected question count differs from the declared cohort')
    if any(not isinstance(row.get('question'), str) or not row['question'].strip() for row in selected):
        raise ValueError('Selected questions need nonblank text')
    predictions = {}
    for line in Path(predictions_path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get('question_id') if isinstance(row, dict) else None
        if not isinstance(qid, str) or not qid.strip() or qid in predictions:
            raise ValueError('Predictions need distinct nonblank question IDs')
        predictions[qid] = row
    if not set(ids) <= predictions.keys():
        raise ValueError('A selected question has no retrieval prediction')
    benchmark = Path(benchmark)
    index = json.loads((benchmark / 'generated_data/uuid_index.json').read_text(encoding='utf-8'))
    source_root = benchmark / 'generated_data/sources'
    cases = []
    for question in selected:
        prediction = predictions[question['question_id']]
        if 'error' not in prediction or prediction['error'] is not None:
            raise ValueError('A selected prediction is missing success metadata or contains a retrieval error')
        doc_ids = prediction.get('retrieved_doc_ids')
        if (not isinstance(doc_ids, list) or any(not isinstance(doc, str) for doc in doc_ids)
                or len(set(doc_ids)) != len(doc_ids)):
            raise ValueError('Retrieved document IDs must be a distinct string list')
        evidence = []
        for rank, doc_id in enumerate(doc_ids[:top_k], start=1):
            relative = index.get(doc_id)
            if not isinstance(relative, str):
                raise ValueError('A retrieved document is missing from the source index')
            document = json.loads(_source_path(source_root, relative).read_text(encoding='utf-8'))
            if document.get('dataset_doc_uuid') != doc_id:
                raise ValueError('Document index points at a different document')
            title, content = extract_document(document)
            evidence.append({'citation_id': f'source-{rank}', 'doc_id': doc_id,
                'source_type': relative.replace('\\', '/').split('/')[0],
                'title': title, 'text': content, 'retrieval_rank': rank})
        # Reference fields and relevant/source labels are never copied into cases.
        cases.append({'question_id': question['question_id'],
            'question_type': question.get('question_type', 'unknown'),
            'question': question['question'], 'evidence': evidence})
    return cases


def write_cases(output: Path, cases: list[dict]) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(cases, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=ROOT / 'outputs/alex_evaluation/questions_20.json')
    parser.add_argument('--predictions', type=Path,
        default=ROOT / 'evaluation/results/bm25_full_development/predictions.jsonl')
    parser.add_argument('--benchmark', type=Path, default=CORPUS_DIR / 'EnterpriseRAG-Bench')
    parser.add_argument('--output', type=Path, required=True, help='New file only; existing saved results are preserved')
    parser.add_argument('--expected-count', type=int, default=20)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args(argv)
    cases = prepare_cases(args.selected, args.predictions, args.benchmark,
                          top_k=args.top_k, expected_count=args.expected_count)
    write_cases(args.output, cases)
    print(f'Saved {len(cases)} legacy development cases to {args.output}')


if __name__ == '__main__':
    main()
