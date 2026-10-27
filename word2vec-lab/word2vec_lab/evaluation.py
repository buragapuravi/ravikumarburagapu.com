"""Analogy and labeled-neighborhood evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from word2vec_lab.model import EmbeddingSpace, SkipGramNS

EmbeddingModel = SkipGramNS | EmbeddingSpace


def _normalized_vectors(model: EmbeddingModel) -> np.ndarray:
    norms = np.linalg.norm(model.vectors, axis=1, keepdims=True)
    return model.vectors / np.maximum(norms, 1e-12)


def nearest_neighbors(
    model: EmbeddingModel, word: str, *, k: int = 10
) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("k must be positive")
    word_id = model.word_to_id.get(word.casefold())
    if word_id is None:
        return []
    normalized = _normalized_vectors(model)
    scores = normalized @ normalized[word_id]
    scores[word_id] = -np.inf
    order = np.argsort(-scores, kind="stable")[: min(k, len(model.words) - 1)]
    return [(model.words[int(index)], float(scores[index])) for index in order]


def evaluate_analogies(
    model: EmbeddingModel, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = _normalized_vectors(model)
    rows = []
    for case in cases:
        terms = [case[key].casefold() for key in ("a", "b", "c", "expected")]
        missing = [term for term in terms if term not in model.word_to_id]
        if missing:
            rows.append(
                {
                    **case,
                    "prediction": None,
                    "correct": None,
                    "status": "oov",
                    "missing_words": missing,
                }
            )
            continue
        a, b, c, _expected = (model.word_to_id[word] for word in terms)
        query = normalized[b] - normalized[a] + normalized[c]
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = normalized @ query
        scores[[a, b, c]] = -np.inf
        prediction = model.words[int(np.argmax(scores))]
        rows.append(
            {
                **case,
                "prediction": prediction,
                "correct": prediction == terms[3],
                "status": "evaluated",
                "missing_words": [],
            }
        )
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    categories = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [row for row in evaluated if row["category"] == category]
        categories[category] = {
            "evaluated": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "accuracy": (
                sum(row["correct"] for row in subset) / len(subset) if subset else None
            ),
        }
    return {
        "total": len(cases),
        "evaluated": len(evaluated),
        "coverage": len(evaluated) / len(cases) if cases else None,
        "correct": sum(row["correct"] for row in evaluated),
        "accuracy": (
            sum(row["correct"] for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "by_category": categories,
        "cases": rows,
    }


def evaluate_neighbors(
    model: EmbeddingModel, cases: list[dict[str, Any]], *, k: int = 5
) -> dict[str, Any]:
    results = []
    for case in cases:
        relevant = {word.casefold() for word in case["relevant"]}
        neighbors = nearest_neighbors(model, case["word"], k=k)
        hits = [word for word, _ in neighbors if word in relevant]
        results.append(
            {
                **case,
                "neighbors": [{"word": word, "similarity": score} for word, score in neighbors],
                "hits": hits,
                "precision_at_k": len(hits) / k,
                "status": "evaluated" if case["word"].casefold() in model.word_to_id else "oov",
            }
        )
    evaluated = [row for row in results if row["status"] == "evaluated"]
    categories = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [row for row in evaluated if row["category"] == category]
        categories[category] = {
            "evaluated": len(subset),
            "precision_at_k": (
                sum(row["precision_at_k"] for row in subset) / len(subset)
                if subset
                else None
            ),
        }
    return {
        "total": len(cases),
        "evaluated": len(evaluated),
        "k": k,
        "precision_at_k": (
            sum(row["precision_at_k"] for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "by_category": categories,
        "cases": results,
    }
