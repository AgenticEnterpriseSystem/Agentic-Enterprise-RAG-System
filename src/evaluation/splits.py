"""Split questions while keeping shared references and duplicate questions together."""

import random
from collections import Counter, defaultdict


def build_splits(questions: list[dict], seed: int = 5980) -> dict:
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Questions must have unique IDs and be nonempty")
    questions = sorted(questions, key=lambda q: q["question_id"])
    # Union questions sharing reference documents OR identical question text.
    parent = list(range(len(questions)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, q in enumerate(questions):
        keys = [("doc", d) for d in q["expected_doc_ids"]]
        keys.append(("text", " ".join(q["question"].lower().split())))
        for key in keys:
            if key in owners:
                parent[root(i)] = root(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, q in enumerate(questions):
        groups.setdefault(root(i), []).append(q)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    fractions = {"development": 0.6, "calibration": 0.2, "final": 0.2}
    totals = Counter(q["question_type"] for q in questions)
    counts = {name: Counter() for name in fractions}
    output = {name: [] for name in fractions}
    for group in groups:
        addition = Counter(q["question_type"] for q in group)

        def balance_cost(name):
            cost = 0.0
            for category, count in addition.items():
                target = totals[category] * fractions[name]
                before = counts[name][category] - target
                after = before + count
                cost += (after**2 - before**2) / max(target, 1)
            return cost

        name = min(fractions, key=balance_cost)
        output[name].extend(q["question_id"] for q in group)
        counts[name].update(addition)
    for values in output.values():
        values.sort()
    # Small regression set: up to six development examples per question type.
    dev_ids = set(output["development"])
    golden = []
    for category in sorted(totals):
        candidates = [
            q["question_id"]
            for q in questions
            if q["question_type"] == category and q["question_id"] in dev_ids
        ]
        random.Random(f"{seed}:{category}").shuffle(candidates)
        golden.extend(candidates[:6])
    return {
        "version": 1,
        "seed": seed,
        "method": "reference-document/duplicate-text groups; greedy category balance 60/20/20",
        "splits": output,
        "slice_counts": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "golden_development_ids": sorted(golden),
    }


def select_questions(questions, count):
    """Deterministic category round-robin, independent of model outcomes."""
    if not 1 <= count <= len(questions):
        raise ValueError("Question count must fit the development set")
    groups = defaultdict(list)
    for q in sorted(questions, key=lambda q: q['question_id']):
        groups[q['question_type']].append(q)
    selected = []
    while len(selected) < count:
        for category in sorted(groups):
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop(0))
    return selected
