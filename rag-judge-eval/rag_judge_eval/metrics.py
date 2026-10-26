"""Agreement metrics for categorical answer and citation judgments."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

LABELS = ("correct", "partially_correct", "wrong")


def cohens_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right):
        raise ValueError("paired label lists must have equal length")
    if not left:
        return None
    if any(value not in LABELS for value in left + right):
        raise ValueError("answer labels must be correct, partially_correct, or wrong")

    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        left_counts[label] * right_counts[label] for label in LABELS
    ) / (len(left) ** 2)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def binary_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right):
        raise ValueError("paired label lists must have equal length")
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = (
        sum(left) * sum(right) + (len(left) - sum(left)) * (len(right) - sum(right))
    ) / (len(left) ** 2)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _agreement(left: list[Any], right: list[Any]) -> float | None:
    return sum(a == b for a, b in zip(left, right)) / len(left) if left else None


def summarize_prompt(rows: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    paired = [
        row
        for row in rows
        if row.get("human_answer_label") in LABELS
        and row.get("human_citation_supported") in (True, False)
        and isinstance(row.get(prompt), dict)
    ]
    human_answers = [row["human_answer_label"] for row in paired]
    judged_answers = [row[prompt]["answer_label"] for row in paired]
    human_citations = [row["human_citation_supported"] for row in paired]
    judged_citations = [row[prompt]["citation_supported"] for row in paired]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_category[row["question_type"]].append(row)

    categories = {}
    for category, category_rows in sorted(by_category.items()):
        human = [row["human_answer_label"] for row in category_rows]
        judged = [row[prompt]["answer_label"] for row in category_rows]
        answer_disagreements = sum(a != b for a, b in zip(human, judged))
        citation_disagreements = sum(
            row["human_citation_supported"] != row[prompt]["citation_supported"]
            for row in category_rows
        )
        categories[category] = {
            "n": len(category_rows),
            "answer_agreement": _agreement(human, judged),
            "answer_kappa": cohens_kappa(human, judged),
            "answer_disagreements": answer_disagreements,
            "citation_agreement": _agreement(
                [row["human_citation_supported"] for row in category_rows],
                [row[prompt]["citation_supported"] for row in category_rows],
            ),
            "citation_disagreements": citation_disagreements,
        }
    return {
        "n_labeled": len(paired),
        "answer_agreement": _agreement(human_answers, judged_answers),
        "answer_kappa": cohens_kappa(human_answers, judged_answers),
        "citation_agreement": _agreement(human_citations, judged_citations),
        "citation_kappa": binary_kappa(human_citations, judged_citations),
        "answer_confusion": {
            human: {
                judged: sum(
                    row["human_answer_label"] == human
                    and row[prompt]["answer_label"] == judged
                    for row in paired
                )
                for judged in LABELS
            }
            for human in LABELS
        },
        "by_question_type": categories,
    }
