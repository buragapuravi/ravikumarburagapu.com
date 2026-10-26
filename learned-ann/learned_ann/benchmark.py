"""Recall, QPS, build-time, and approximate memory measurement."""

import time
from typing import Any

import numpy as np

from learned_ann.indexes import exact_search, make_index


def recall_at_k(expected: Any, actual: Any, k: int = 10) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    gold = np.asarray(expected)
    found = np.asarray(actual)
    if gold.ndim != 2 or found.ndim != 2 or gold.shape[0] != found.shape[0]:
        raise ValueError("expected and actual IDs must be 2D arrays with matching row counts")
    if gold.shape[0] == 0:
        raise ValueError("cannot evaluate recall on an empty query set")
    scores = []
    for expected_row, actual_row in zip(gold, found, strict=True):
        expected_set = {int(item) for item in expected_row[:k] if int(item) >= 0}
        actual_set = {int(item) for item in actual_row[:k] if int(item) >= 0}
        scores.append(len(expected_set & actual_set) / max(1, len(expected_set)))
    return float(np.mean(scores))


def pareto_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated runs maximizing recall and QPS."""
    frontier = []
    for candidate in results:
        dominated = any(
            other is not candidate
            and other["recall_at_k"] >= candidate["recall_at_k"]
            and other["qps"] >= candidate["qps"]
            and (
                other["recall_at_k"] > candidate["recall_at_k"]
                or other["qps"] > candidate["qps"]
            )
            for other in results
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item["qps"], item["recall_at_k"]))


def benchmark_index(
    name: str,
    vectors: Any,
    queries: Any,
    *,
    k: int = 10,
    metric: str = "cosine",
    index_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = np.asarray(vectors, dtype=np.float32)
    query_vectors = np.asarray(queries, dtype=np.float32)
    build_started = time.perf_counter()
    index = make_index(name, metric, **(index_options or {}))
    index.fit(data)
    measured_build_ms = (time.perf_counter() - build_started) * 1000
    expected, _ = exact_search(data, query_vectors, k=k, metric=metric)
    started = time.perf_counter()
    actual, _ = index.search(query_vectors, k=k)
    elapsed = time.perf_counter() - started
    return {
        "index": name,
        "metric": metric,
        "recall_at_k": recall_at_k(expected, actual, k),
        "k": k,
        "query_count": len(query_vectors),
        "qps": len(query_vectors) / elapsed if elapsed > 0 else 0.0,
        "query_latency_ms": elapsed * 1000 / len(query_vectors) if len(query_vectors) else 0.0,
        "build_time_ms": measured_build_ms,
        "memory_bytes_estimate": index.memory_bytes(),
        "index_options": index_options or {},
        "note": _memory_note(name),
    }


def _memory_note(name: str) -> str:
    if name == "hnsw":
        return "Estimate: stored float vectors plus a rough link-array allowance; not allocator RSS."
    if name == "scann":
        return "Estimate: 1.15x the stored float-vector bytes; not measured native allocator RSS."
    if name == "faiss-ivf-pq":
        return "Analytical estimate for Faiss PQ codes, centroids, and codebooks; excludes native overhead and allocator RSS."
    return "Tracked NumPy arrays only; excludes Python, SQLite/runtime, and allocator overhead."


def synthetic_data(
    vector_count: int = 4000,
    query_count: int = 160,
    dimension: int = 32,
    cluster_count: int = 20,
    seed: int = 41,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if vector_count < 20 or query_count < 1 or dimension < 4 or cluster_count < 1:
        raise ValueError("need >=20 vectors, >=1 query, dimension >=4, and >=1 cluster")
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(cluster_count, dimension)).astype(np.float32)
    assignments = rng.integers(0, cluster_count, size=vector_count)
    vectors = centers[assignments] + 0.28 * rng.normal(size=(vector_count, dimension))
    query_clusters = rng.integers(0, cluster_count, size=query_count)
    queries = centers[query_clusters] + 0.28 * rng.normal(size=(query_count, dimension))
    return vectors.astype(np.float32), queries.astype(np.float32), query_clusters
