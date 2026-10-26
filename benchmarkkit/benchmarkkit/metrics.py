"""Deterministic summary and retrieval/answer evaluation metrics."""

import math
import re
from collections.abc import Iterable
from typing import Any


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize_inference(
    results: list[dict[str, Any]], elapsed_seconds: float, input_price: float, output_price: float
) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty inference run")
    total_tokens = sum(result["input_tokens"] + result["output_tokens"] for result in results)
    input_tokens = sum(result["input_tokens"] for result in results)
    output_tokens = sum(result["output_tokens"] for result in results)
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return {
        "kind": "inference",
        "request_count": len(results),
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": len(results) / elapsed_seconds if elapsed_seconds > 0 else 0,
        "output_tokens_per_second": output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0,
        "ttft_ms": {
            "p50": percentile((r["ttft_ms"] for r in results), 0.50),
            "p95": percentile((r["ttft_ms"] for r in results), 0.95),
            "p99": percentile((r["ttft_ms"] for r in results), 0.99),
        },
        "latency_ms": {
            "p50": percentile((r["latency_ms"] for r in results), 0.50),
            "p95": percentile((r["latency_ms"] for r in results), 0.95),
            "p99": percentile((r["latency_ms"] for r in results), 0.99),
        },
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0,
        "cost_usd": cost,
        "cost_per_1k_tokens_usd": cost * 1000 / total_tokens if total_tokens else 0,
        "requests": results,
    }


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with",
}


def _terms(text: str) -> set[str]:
    return {word for word in _TOKEN.findall(text.lower()) if word not in _STOPWORDS}


def evaluate_rag_record(record: dict[str, Any], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    expected = record.get("expected_doc_ids")
    retrieved = record.get("retrieved_doc_ids")
    answer = record.get("answer")
    contexts = record.get("contexts")
    if not isinstance(expected, list) or not expected:
        raise ValueError("'expected_doc_ids' must be a non-empty list")
    if not isinstance(retrieved, list):
        raise ValueError("'retrieved_doc_ids' must be a list")
    if not isinstance(answer, str):
        raise ValueError("'answer' must be a string")
    if not isinstance(contexts, list) or not all(isinstance(context, str) for context in contexts):
        raise ValueError("'contexts' must be a list of strings")
    if any(k < 1 for k in ks):
        raise ValueError("recall cutoffs must be positive")

    answer_terms = _terms(answer)
    context_terms = _terms(" ".join(contexts))
    return {
        "recall": {
            f"recall_at_{k}": len(set(expected).intersection(retrieved[:k])) / len(set(expected))
            for k in ks
        },
        "faithfulness_proxy": (
            len(answer_terms.intersection(context_terms)) / len(answer_terms) if answer_terms else 0
        ),
        "latency_seconds": _non_negative_number(record.get("latency_seconds", 0), "latency_seconds"),
    }


def summarize_rag(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty RAG run")
    evaluated = [evaluate_rag_record(record) for record in records]
    recall_keys = evaluated[0]["recall"]
    return {
        "kind": "rag",
        "record_count": len(evaluated),
        "recall": {
            key: sum(item["recall"][key] for item in evaluated) / len(evaluated)
            for key in recall_keys
        },
        "faithfulness_proxy": sum(item["faithfulness_proxy"] for item in evaluated) / len(evaluated),
        "latency_ms": {
            "p50": percentile((item["latency_seconds"] * 1000 for item in evaluated), 0.50),
            "p95": percentile((item["latency_seconds"] * 1000 for item in evaluated), 0.95),
            "p99": percentile((item["latency_seconds"] * 1000 for item in evaluated), 0.99),
        },
        "records": evaluated,
    }


def _non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"'{name}' must be a non-negative number")
    return float(value)
