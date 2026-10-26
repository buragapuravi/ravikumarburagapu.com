"""Retrieval and answer evaluation helpers."""

from __future__ import annotations

import re
from typing import Any

TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(TOKEN.findall(text.casefold()))


def recall_at_k(ranked_chunks: list[Any], evidence: str, k: int = 5) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    target = normalize(evidence)
    return float(any(target in normalize(chunk.text) for chunk in ranked_chunks[:k]))


def exact_answer_accuracy(answer: str, expected_answer: str) -> float:
    return float(normalize(answer) == normalize(expected_answer))


def required_phrase_coverage(answer: str, expected_answer: str) -> float:
    required = set(TOKEN.findall(expected_answer.casefold()))
    if not required:
        return 0.0
    present = set(TOKEN.findall(answer.casefold()))
    return len(required & present) / len(required)


def context_token_cost(
    contexts: list[tuple[str, int]],
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
    estimated_output_tokens: int = 0,
) -> dict[str, float]:
    if input_cost_per_million < 0 or output_cost_per_million < 0 or estimated_output_tokens < 0:
        raise ValueError("token costs and output tokens cannot be negative")
    total = sum(tokens for _, tokens in contexts)
    query_count = len(contexts)
    avg = total / query_count if query_count else 0.0
    cost = (
        avg * input_cost_per_million
        + estimated_output_tokens * output_cost_per_million
    ) / 1_000_000
    return {
        "avg_context_tokens": avg,
        "avg_estimated_cost_per_query": cost,
        "input_cost_per_million": input_cost_per_million,
        "output_cost_per_million": output_cost_per_million,
        "estimated_output_tokens": float(estimated_output_tokens),
    }
