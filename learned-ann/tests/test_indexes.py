import unittest

import numpy as np

from learned_ann.benchmark import benchmark_index, pareto_frontier, recall_at_k, synthetic_data
from learned_ann.indexes import (
    IVFPQIndex,
    LearnedRMIIndex,
    exact_search,
    make_index,
    prepare_vectors,
)


class IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors, self.queries, _ = synthetic_data(
            vector_count=240,
            query_count=12,
            dimension=8,
            cluster_count=6,
            seed=5,
        )

    def test_cosine_vectors_are_normalized(self) -> None:
        normalized = prepare_vectors(self.vectors)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1, atol=1e-6)

    def test_rmi_search_returns_valid_neighbors_and_has_tree(self) -> None:
        index = LearnedRMIIndex(branch_factor=3, max_leaf_size=24, nprobe=3).fit(self.vectors)
        ids, distances = index.search(self.queries, k=5)
        self.assertEqual(ids.shape, (len(self.queries), 5))
        self.assertEqual(distances.shape, (len(self.queries), 5))
        self.assertTrue(np.all(ids >= 0))
        self.assertGreater(len(index._leaf_nodes), 1)
        self.assertGreater(index.memory_bytes(), self.vectors.nbytes)

    def test_rmi_probing_all_partitions_matches_exact(self) -> None:
        index = LearnedRMIIndex(
            branch_factor=3,
            max_leaf_size=24,
            nprobe=len(self.vectors),
        ).fit(self.vectors)
        expected, _ = exact_search(self.vectors, self.queries, k=5)
        actual, _ = index.search(self.queries, k=5)
        routed_leaves = index._route(prepare_vectors(self.queries[:1])[0])
        self.assertEqual(len(routed_leaves), len(index._leaf_nodes))
        self.assertEqual(recall_at_k(expected, actual, 5), 1)

    def test_l2_exact_and_index_search(self) -> None:
        expected, _ = exact_search(self.vectors, self.queries, k=5, metric="l2")
        index = LearnedRMIIndex(
            branch_factor=3,
            max_leaf_size=24,
            nprobe=len(self.vectors),
            metric="l2",
        ).fit(self.vectors)
        actual, distances = index.search(self.queries, k=5)
        self.assertEqual(recall_at_k(expected, actual, 5), 1)
        self.assertTrue(np.all(distances[:, 1:] >= distances[:, :-1]))

    def test_ivfpq_smoke_search_and_dimension_guard(self) -> None:
        index = IVFPQIndex(nlist=8, nprobe=8, subquantizers=2, bits=4).fit(self.vectors)
        ids, distances = index.search(self.queries, k=5)
        self.assertEqual(ids.shape, (len(self.queries), 5))
        self.assertTrue(np.isfinite(distances).all())
        with self.assertRaisesRegex(ValueError, "divisible"):
            IVFPQIndex(subquantizers=3).fit(self.vectors)

    def test_recall_and_benchmark_report_metrics(self) -> None:
        expected, _ = exact_search(self.vectors, self.queries, k=5)
        self.assertEqual(recall_at_k(expected, expected, 5), 1)
        result = benchmark_index(
            "learned-rmi",
            self.vectors,
            self.queries,
            k=5,
            index_options={"branch_factor": 3, "max_leaf_size": 24, "nprobe": 8},
        )
        for metric in ("recall_at_k", "qps", "build_time_ms", "memory_bytes_estimate"):
            self.assertIn(metric, result)
        self.assertEqual(result["recall_at_k"], 1)

    def test_invalid_vectors_and_unknown_indexes_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_vectors([[float("nan")]])
        with self.assertRaisesRegex(ValueError, "one of"):
            make_index("unknown")

    def test_pareto_frontier_removes_dominated_runs(self) -> None:
        runs = [
            {"index": "fast", "recall_at_k": 0.8, "qps": 100},
            {"index": "accurate", "recall_at_k": 0.95, "qps": 40},
            {"index": "dominated", "recall_at_k": 0.7, "qps": 30},
        ]
        self.assertEqual(
            {run["index"] for run in pareto_frontier(runs)},
            {"fast", "accurate"},
        )

    def test_pareto_frontier_keeps_identical_measurements(self) -> None:
        runs = [
            {"index": "run-a", "recall_at_k": 0.9, "qps": 50},
            {"index": "run-b", "recall_at_k": 0.9, "qps": 50},
        ]
        self.assertEqual(len(pareto_frontier(runs)), 2)


if __name__ == "__main__":
    unittest.main()
