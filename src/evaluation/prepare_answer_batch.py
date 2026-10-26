"""Prepare BM25+ evidence for our selected 20 evaluation questions."""

import json
from pathlib import Path
from src.config import ROOT, CORPUS_DIR
from src.data.corpus import extract_document


def main():
    benchmark = CORPUS_DIR / "EnterpriseRAG-Bench"
    folder = ROOT / "outputs" / "alex_evaluation"

    # Load our selected 20 development questions.
    selected = json.loads(
        (folder / "questions_20.json").read_text(encoding="utf-8")
    )

    # Load the team's existing BM25+ retrieval predictions.
    predictions_path = (
        ROOT
        / "evaluation"
        / "results"
        / "bm25_full_development"
        / "predictions.jsonl"
    )

    predictions = {
        row["question_id"]: row
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))
    }

    # Map document IDs to their stored source files.
    index = json.loads(
        (benchmark / "generated_data" / "uuid_index.json")
        .read_text(encoding="utf-8")
    )

    # Verify the selected questions before processing.
    assert len(selected) == 20
    assert len({q["question_id"] for q in selected}) == 20
    assert all(q["question_id"] in predictions for q in selected)

    cases = []

    for question in selected:
        prediction = predictions[question["question_id"]]

        if prediction["error"]:
            raise ValueError(
                f"Retrieval error for {question['question_id']}"
            )

        evidence = []

        # Extract the top five retrieved documents.
        for rank, doc_id in enumerate(
            prediction["retrieved_doc_ids"][:5], start=1
        ):
            path = benchmark / "generated_data" / "sources" / index[doc_id]
            # Use Windows extended-length paths for deeply nested corpus files.
            read_path = Path("\\\\?\\" + str(path.resolve())) if path.is_absolute() or path.exists() else Path("\\\\?\\" + str(path.absolute()))
            document = json.loads(read_path.read_text(encoding="utf-8"))

            title, content = extract_document(document)
            source_type = index[doc_id].replace("\\", "/").split("/")[0]
            evidence.append({
                "citation_id": f"source-{rank}",
                "doc_id": doc_id,
                "source_type": source_type,
                "title": title,
                "text": content,
                "retrieval_rank": rank,
            })

        cases.append({
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "question": question["question"],
            "evidence": evidence,
        })

    output = folder / "bm25_cases_20.json"

    # Create a new file; refuse to overwrite an existing one.
    with output.open("w", encoding="utf-8") as file:
        json.dump(cases, file, indent=2)

    print(f"Saved {len(cases)} cases to {output}")
    print(
        "Total evidence documents:",
        sum(len(case["evidence"]) for case in cases)
    )


if __name__ == "__main__":
    main()