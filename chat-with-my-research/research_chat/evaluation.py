"""Citation and retrieval evaluation for labeled research questions."""

import re
from collections.abc import Iterable
from typing import Any

from research_chat.rag import _CITATION


def evaluate_retrieval(
    expected_source_ids: Iterable[str], retrieved_source_ids: Iterable[str], k: int = 5
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    expected = set(expected_source_ids)
    retrieved = list(retrieved_source_ids)[:k]
    if not expected:
        raise ValueError("expected_source_ids must not be empty")
    def is_expected(source_id: str) -> bool:
        return any(source_id == gold or gold.endswith("-") and source_id.startswith(gold) for gold in expected)

    hits = {source for source in retrieved if is_expected(source)}
    return {
        f"recall_at_{k}": len(hits) / len(expected),
        f"precision_at_{k}": len(hits) / len(retrieved) if retrieved else 0.0,
        f"reciprocal_rank_at_{k}": next(
            (1 / rank for rank, source in enumerate(retrieved, start=1) if is_expected(source)),
            0.0,
        ),
    }


def evaluate_citations(answer: str, source_ids: Iterable[str]) -> dict[str, float | int]:
    allowed = set(source_ids)
    cited_numbers = {int(match) for match in _CITATION.findall(answer)}
    cited = {f"S{number}" for number in cited_numbers}
    return {
        "citation_count": len(cited),
        "unsupported_citation_count": len(cited - allowed),
        "citation_precision": len(cited & allowed) / len(cited) if cited else 0.0,
        "citation_coverage": 1.0 if cited and not (cited - allowed) else 0.0,
    }


def evaluate_answer_coverage(answer: str, required_terms: Iterable[str]) -> float:
    required = {term.casefold() for term in required_terms if term.strip()}
    if not required:
        raise ValueError("required_terms must contain at least one term")
    answer_words = set(re.findall(r"\w+", answer.casefold()))
    required_words = {word for term in required for word in re.findall(r"\w+", term)}
    return len(required_words & answer_words) / len(required_words) if required_words else 0.0


def evaluate_example(
    graph: Any, example: dict[str, Any], k: int = 5
) -> dict[str, Any]:
    question = example.get("question")
    expected_sources = example.get("expected_sources")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("each evaluation example needs a non-empty question")
    if not isinstance(expected_sources, list) or not expected_sources:
        raise ValueError(f"evaluation question {question!r} needs expected_sources labels")
    result = graph.invoke({"question": question})
    sources = result.get("sources", [])
    source_ids = [source["source_id"] for source in sources]
    metrics = evaluate_retrieval(expected_sources, source_ids, k)
    metrics.update(
        evaluate_citations(
            result["answer"],
            (source["citation"] for source in sources),
        )
    )
    required_terms = example.get("required_terms", [])
    if required_terms:
        metrics["answer_coverage"] = evaluate_answer_coverage(result["answer"], required_terms)
    return {"question": question, "metrics": metrics}


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("no evaluation examples were provided")
    keys = evaluations[0]["metrics"]
    return {
        "example_count": len(evaluations),
        "metrics": {
            key: sum(item["metrics"][key] for item in evaluations) / len(evaluations)
            for key in keys
        },
        "examples": evaluations,
    }
