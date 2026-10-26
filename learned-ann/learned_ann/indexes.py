"""Learned RMI partitioning and common ANN index implementations."""

from __future__ import annotations

import heapq
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int64]


def prepare_vectors(vectors: Any, metric: str = "cosine") -> FloatMatrix:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("vectors must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError("vectors must contain only finite values")
    if metric not in {"cosine", "l2"}:
        raise ValueError("metric must be 'cosine' or 'l2'")
    array = np.ascontiguousarray(array)
    if metric == "cosine":
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("cosine metric does not support zero vectors")
        array = np.ascontiguousarray(array / norms)
    return array


def _distance(query: FloatMatrix, vectors: FloatMatrix, metric: str) -> FloatMatrix:
    if metric == "cosine":
        norms = np.linalg.norm(vectors, axis=1)
        similarities = vectors @ query.reshape(-1)
        similarities = np.divide(similarities, norms, out=np.zeros_like(similarities), where=norms > 0)
        return 1.0 - similarities
    delta = vectors - query
    return np.einsum("ij,ij->i", delta, delta, optimize=True)


def _validate_queries(queries: Any, dimension: int, metric: str) -> FloatMatrix:
    result = prepare_vectors(queries, metric)
    if result.shape[1] != dimension:
        raise ValueError(f"query dimension {result.shape[1]} does not match index dimension {dimension}")
    return result


def _top_k(ids: NDArray[np.integer], distances: FloatMatrix, k: int) -> tuple[IntVector, FloatMatrix]:
    if k < 1:
        raise ValueError("k must be positive")
    count = min(k, len(ids))
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    positions = np.argpartition(distances, count - 1)[:count]
    positions = positions[np.argsort(distances[positions], kind="stable")]
    return np.asarray(ids[positions], dtype=np.int64), np.asarray(distances[positions], dtype=np.float32)


def _kmeans(
    vectors: FloatMatrix,
    cluster_count: int,
    iterations: int = 18,
    seed: int = 17,
) -> tuple[FloatMatrix, IntVector]:
    if cluster_count < 1:
        raise ValueError("cluster_count must be positive")
    count = len(vectors)
    cluster_count = min(cluster_count, count)
    rng = np.random.default_rng(seed)
    centers = np.empty((cluster_count, vectors.shape[1]), dtype=np.float32)
    centers[0] = vectors[rng.integers(count)]
    closest = np.sum((vectors - centers[0]) ** 2, axis=1)
    for center_index in range(1, cluster_count):
        total = float(closest.sum())
        if total > 0:
            selected = int(rng.choice(count, p=closest / total))
        else:
            selected = center_index % count
        centers[center_index] = vectors[selected]
        closest = np.minimum(closest, np.sum((vectors - centers[center_index]) ** 2, axis=1))

    labels = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        distances = (
            np.sum(vectors**2, axis=1, keepdims=True)
            + np.sum(centers**2, axis=1)[None, :]
            - 2 * vectors @ centers.T
        )
        next_labels = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(labels, next_labels) and _:
            break
        labels = next_labels
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, vectors)
        counts = np.bincount(labels, minlength=cluster_count)
        active = counts > 0
        centers[active] = sums[active] / counts[active, None]
        for empty in np.flatnonzero(~active):
            largest = int(np.argmax(np.min(distances, axis=1)))
            centers[empty] = vectors[largest]
    return centers, labels


class ANNIndex(ABC):
    """Shared approximate-index interface."""

    name: str

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.metric = metric
        self.vectors: FloatMatrix | None = None
        self.build_time_ms = 0.0

    @abstractmethod
    def fit(self, vectors: Any) -> "ANNIndex":
        raise NotImplementedError

    @abstractmethod
    def search(self, queries: Any, k: int = 10) -> tuple[IntVector, FloatMatrix]:
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self) -> int:
        raise NotImplementedError

    def _set_vectors(self, vectors: Any) -> FloatMatrix:
        self.vectors = prepare_vectors(vectors, self.metric)
        return self.vectors


@dataclass
class RMINode:
    center: FloatMatrix
    depth: int
    ids: IntVector | None = None
    children: list["RMINode"] = field(default_factory=list)
    radius: float = 0.0


class LearnedRMIIndex(ANNIndex):
    """Recursive k-means partition model with exact or local-HNSW leaf search."""

    name = "learned-rmi"

    def __init__(
        self,
        branch_factor: int = 4,
        max_leaf_size: int = 128,
        nprobe: int = 2,
        metric: str = "cosine",
        inner_backend: str = "exact",
        hnsw_m: int = 16,
        ef_search: int = 64,
        seed: int = 17,
    ) -> None:
        super().__init__(metric)
        if branch_factor < 2 or max_leaf_size < 1 or nprobe < 1:
            raise ValueError("branch_factor >= 2, max_leaf_size >= 1 and nprobe >= 1 are required")
        if inner_backend not in {"exact", "hnsw"}:
            raise ValueError("inner_backend must be 'exact' or 'hnsw'")
        self.branch_factor = branch_factor
        self.max_leaf_size = max_leaf_size
        self.nprobe = nprobe
        self.inner_backend = inner_backend
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.seed = seed
        self.root: RMINode | None = None
        self._leaf_nodes: list[RMINode] = []
        self._hnsw_by_leaf: dict[int, Any] = {}

    def fit(self, vectors: Any) -> "LearnedRMIIndex":
        started = time.perf_counter()
        data = self._set_vectors(vectors)
        ids = np.arange(len(data), dtype=np.int64)
        self._leaf_nodes = []
        self._hnsw_by_leaf = {}
        self.root = self._build_node(data, ids, 0)
        if self.inner_backend == "hnsw":
            try:
                import hnswlib
            except ImportError as error:
                raise RuntimeError("Install the HNSW extra with `pip install -e '.[hnsw]'`.") from error
            for leaf_number, leaf in enumerate(self._leaf_nodes):
                if leaf.ids is None or len(leaf.ids) < 2:
                    continue
                index = hnswlib.Index(space="cosine" if self.metric == "cosine" else "l2", dim=data.shape[1])
                index.init_index(max_elements=len(leaf.ids), ef_construction=100, M=self.hnsw_m)
                index.add_items(data[leaf.ids], leaf.ids)
                index.set_ef(max(self.ef_search, self.nprobe))
                self._hnsw_by_leaf[leaf_number] = index
        self.build_time_ms = (time.perf_counter() - started) * 1000
        return self

    def _build_node(self, vectors: FloatMatrix, ids: IntVector, depth: int) -> RMINode:
        center = vectors.mean(axis=0).astype(np.float32)
        radius = float(np.max(np.linalg.norm(vectors - center, axis=1))) if len(vectors) else 0.0
        node = RMINode(center=center, depth=depth, radius=radius)
        if len(ids) <= self.max_leaf_size:
            node.ids = ids.copy()
            self._leaf_nodes.append(node)
            return node
        cluster_count = min(self.branch_factor, max(2, math.ceil(len(ids) / self.max_leaf_size)))
        centers, labels = _kmeans(
            vectors,
            cluster_count,
            seed=self.seed + depth * 1009 + int(ids[0]),
        )
        for cluster_id in range(cluster_count):
            mask = labels == cluster_id
            if np.any(mask):
                node.children.append(self._build_node(vectors[mask], ids[mask], depth + 1))
        if len(node.children) < 2:
            node.ids = ids.copy()
            node.children = []
            self._leaf_nodes.append(node)
        return node

    def _leaf_distances(self, query: FloatMatrix) -> list[tuple[float, int, RMINode]]:
        ranked: list[tuple[float, int, RMINode]] = []
        for index, leaf in enumerate(self._leaf_nodes):
            distance = float(_distance(query[None, :], leaf.center[None, :], self.metric)[0])
            ranked.append((distance, index, leaf))
        return sorted(ranked, key=lambda item: item[0])

    def _route(self, query: FloatMatrix) -> list[tuple[float, int, RMINode]]:
        if self.root is None:
            raise RuntimeError("index is not fitted")
        frontier = [self.root]
        while any(node.children for node in frontier):
            candidates: list[tuple[float, RMINode]] = []
            for node in frontier:
                nodes = node.children or [node]
                for candidate in nodes:
                    distance = float(
                        _distance(query[None, :], candidate.center[None, :], self.metric)[0]
                    )
                    candidates.append((distance, candidate))
            frontier = [
                node
                for _, node in sorted(candidates, key=lambda item: item[0])[: self.nprobe]
            ]
        selected_ids = {id(node) for node in frontier}
        all_leaves = self._leaf_distances(query)
        selected = [item for item in all_leaves if id(item[2]) in selected_ids]
        if len(selected) < self.nprobe:
            selected_node_ids = {id(item[2]) for item in selected}
            selected.extend(
                item for item in all_leaves
                if id(item[2]) not in selected_node_ids
            )
            selected = selected[: self.nprobe]
        return selected

    def search(self, queries: Any, k: int = 10) -> tuple[IntVector, FloatMatrix]:
        if self.vectors is None:
            raise RuntimeError("index is not fitted")
        if k < 1:
            raise ValueError("k must be positive")
        query_vectors = _validate_queries(queries, self.vectors.shape[1], self.metric)
        output_ids = np.full((len(query_vectors), k), -1, dtype=np.int64)
        output_distances = np.full((len(query_vectors), k), np.inf, dtype=np.float32)
        for row, query in enumerate(query_vectors):
            selected = self._route(query)
            candidate_parts: list[IntVector] = []
            for leaf_distance, leaf_number, leaf in selected:
                if leaf.ids is None:
                    continue
                if self.inner_backend == "hnsw" and leaf_number in self._hnsw_by_leaf:
                    found_ids, _ = self._hnsw_by_leaf[leaf_number].knn_query(
                        query, k=min(k, len(leaf.ids))
                    )
                    candidate_parts.append(np.asarray(found_ids[0], dtype=np.int64))
                else:
                    candidate_parts.append(leaf.ids)
            if not candidate_parts:
                continue
            candidates = np.unique(np.concatenate(candidate_parts))
            distances = _distance(query[None, :], self.vectors[candidates], self.metric)
            result_ids, result_distances = _top_k(candidates, distances, k)
            output_ids[row, : len(result_ids)] = result_ids
            output_distances[row, : len(result_distances)] = result_distances
        return output_ids, output_distances

    def memory_bytes(self) -> int:
        if self.vectors is None:
            return 0
        node_bytes = sum(node.center.nbytes + (node.ids.nbytes if node.ids is not None else 0) for node in self._all_nodes())
        hnsw_estimate = (
            sum(
                len(leaf.ids) * (self.hnsw_m * 2 + 8) * 4
                for index, leaf in enumerate(self._leaf_nodes)
                if leaf.ids is not None and index in self._hnsw_by_leaf
            )
            if self.inner_backend == "hnsw"
            else 0
        )
        return int(self.vectors.nbytes + node_bytes + hnsw_estimate)

    def _all_nodes(self) -> list[RMINode]:
        if self.root is None:
            return []
        nodes = [self.root]
        for child in self.root.children:
            nodes.extend(self._descendants(child))
        return nodes

    @staticmethod
    def _descendants(node: RMINode) -> list[RMINode]:
        nodes = [node]
        for child in node.children:
            nodes.extend(LearnedRMIIndex._descendants(child))
        return nodes


class IVFPQIndex(ANNIndex):
    """Coarse inverted-file index with residual product quantization."""

    name = "ivf-pq"

    def __init__(
        self,
        nlist: int = 32,
        nprobe: int = 4,
        subquantizers: int = 8,
        bits: int = 8,
        metric: str = "cosine",
        seed: int = 23,
    ) -> None:
        super().__init__(metric)
        if nlist < 1 or nprobe < 1 or subquantizers < 1:
            raise ValueError("nlist, nprobe, and subquantizers must be positive")
        if bits < 1 or bits > 8:
            raise ValueError("bits must be between 1 and 8")
        self.nlist = nlist
        self.nprobe = nprobe
        self.subquantizers = subquantizers
        self.bits = bits
        self.seed = seed
        self._coarse_centers: FloatMatrix | None = None
        self._codebooks: list[FloatMatrix] = []
        self._codes: NDArray[np.uint8] | None = None
        self._assignments: IntVector | None = None

    def fit(self, vectors: Any) -> "IVFPQIndex":
        started = time.perf_counter()
        data = self._set_vectors(vectors)
        if data.shape[1] % self.subquantizers:
            raise ValueError("embedding dimension must be divisible by subquantizers")
        self._coarse_centers, self._assignments = _kmeans(
            data, min(self.nlist, len(data)), seed=self.seed
        )
        residuals = data - self._coarse_centers[self._assignments]
        subdimension = data.shape[1] // self.subquantizers
        code_count = min(1 << self.bits, len(data))
        codes = np.zeros((len(data), self.subquantizers), dtype=np.uint8)
        self._codebooks = []
        for part in range(self.subquantizers):
            start = part * subdimension
            end = start + subdimension
            centers, labels = _kmeans(
                residuals[:, start:end],
                min(code_count, len(data)),
                seed=self.seed + part * 31,
            )
            self._codebooks.append(centers)
            codes[:, part] = labels.astype(np.uint8)
        self._codes = codes
        self.build_time_ms = (time.perf_counter() - started) * 1000
        return self

    def search(self, queries: Any, k: int = 10) -> tuple[IntVector, FloatMatrix]:
        if self.vectors is None or self._coarse_centers is None or self._assignments is None or self._codes is None:
            raise RuntimeError("index is not fitted")
        if k < 1:
            raise ValueError("k must be positive")
        query_vectors = _validate_queries(queries, self.vectors.shape[1], self.metric)
        output_ids = np.full((len(query_vectors), k), -1, dtype=np.int64)
        output_distances = np.full((len(query_vectors), k), np.inf, dtype=np.float32)
        subdimension = self.vectors.shape[1] // self.subquantizers
        for row, query in enumerate(query_vectors):
            coarse_distances = _distance(query[None, :], self._coarse_centers, self.metric)
            selected_lists = np.argsort(coarse_distances)[: min(self.nprobe, len(coarse_distances))]
            candidates = np.flatnonzero(np.isin(self._assignments, selected_lists))
            if not len(candidates):
                continue
            reconstructed = self._coarse_centers[self._assignments[candidates]].copy()
            for part, codebook in enumerate(self._codebooks):
                start = part * subdimension
                end = start + subdimension
                reconstructed[:, start:end] += codebook[self._codes[candidates, part]]
            approx = _distance(query[None, :], reconstructed, self.metric)
            shortlist_size = min(max(k * 4, k), len(candidates))
            shortlist_positions = np.argpartition(approx, shortlist_size - 1)[:shortlist_size]
            shortlist = candidates[shortlist_positions]
            exact_distances = _distance(query[None, :], self.vectors[shortlist], self.metric)
            result_ids, result_distances = _top_k(shortlist, exact_distances, k)
            output_ids[row, : len(result_ids)] = result_ids
            output_distances[row, : len(result_distances)] = result_distances
        return output_ids, output_distances

    def memory_bytes(self) -> int:
        return int(
            (self.vectors.nbytes if self.vectors is not None else 0)
            + (self._coarse_centers.nbytes if self._coarse_centers is not None else 0)
            + (self._codes.nbytes if self._codes is not None else 0)
            + sum(codebook.nbytes for codebook in self._codebooks)
            + (self._assignments.nbytes if self._assignments is not None else 0)
        )


class HNSWIndex(ANNIndex):
    """Optional hnswlib baseline."""

    name = "hnsw"

    def __init__(self, metric: str = "cosine", m: int = 16, ef_construction: int = 200, ef_search: int = 64):
        super().__init__(metric)
        if m < 2 or ef_construction < 1 or ef_search < 1:
            raise ValueError("HNSW parameters must be positive (m >= 2)")
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: Any = None
        self._hnswlib: Any = None

    def fit(self, vectors: Any) -> "HNSWIndex":
        started = time.perf_counter()
        data = self._set_vectors(vectors)
        try:
            import hnswlib
        except ImportError as error:
            raise RuntimeError("Install HNSW support with `pip install -e '.[hnsw]'`.") from error
        self._hnswlib = hnswlib
        self._index = hnswlib.Index(
            space="cosine" if self.metric == "cosine" else "l2",
            dim=data.shape[1],
        )
        self._index.init_index(max_elements=len(data), ef_construction=self.ef_construction, M=self.m)
        self._index.add_items(data, np.arange(len(data), dtype=np.int64))
        self._index.set_ef(self.ef_search)
        self.build_time_ms = (time.perf_counter() - started) * 1000
        return self

    def search(self, queries: Any, k: int = 10) -> tuple[IntVector, FloatMatrix]:
        if self.vectors is None or self._index is None:
            raise RuntimeError("index is not fitted")
        if k < 1:
            raise ValueError("k must be positive")
        query_vectors = _validate_queries(queries, self.vectors.shape[1], self.metric)
        ids, distances = self._index.knn_query(query_vectors, k=min(k, len(self.vectors)))
        return np.asarray(ids, dtype=np.int64), np.asarray(distances, dtype=np.float32)

    def memory_bytes(self) -> int:
        if self.vectors is None:
            return 0
        graph_bytes = len(self.vectors) * self.m * 2 * 4
        return int(self.vectors.nbytes + graph_bytes)


class FaissIVFPQIndex(ANNIndex):
    """Optional Faiss native IVF-PQ baseline."""

    name = "faiss-ivf-pq"

    def __init__(
        self,
        metric: str = "cosine",
        nlist: int = 64,
        nprobe: int = 8,
        subquantizers: int = 8,
        bits: int = 8,
    ) -> None:
        super().__init__(metric)
        if nlist < 1 or nprobe < 1 or subquantizers < 1 or not 1 <= bits <= 8:
            raise ValueError("nlist/nprobe/subquantizers must be positive and bits must be 1–8")
        self.nlist = nlist
        self.nprobe = nprobe
        self.subquantizers = subquantizers
        self.bits = bits
        self._index: Any = None
        self._faiss: Any = None

    def fit(self, vectors: Any) -> "FaissIVFPQIndex":
        started = time.perf_counter()
        data = self._set_vectors(vectors)
        if data.shape[1] % self.subquantizers:
            raise ValueError("embedding dimension must be divisible by subquantizers")
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("Install Faiss support with `pip install -e '.[faiss]'`.") from error
        self._faiss = faiss
        metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        quantizer = (
            faiss.IndexFlatIP(data.shape[1])
            if self.metric == "cosine"
            else faiss.IndexFlatL2(data.shape[1])
        )
        self._index = faiss.IndexIVFPQ(
            quantizer,
            data.shape[1],
            min(self.nlist, len(data)),
            self.subquantizers,
            self.bits,
            metric_type,
        )
        self._index.train(data)
        self._index.add(data)
        self._index.nprobe = min(self.nprobe, self._index.nlist)
        self.build_time_ms = (time.perf_counter() - started) * 1000
        return self

    def search(self, queries: Any, k: int = 10) -> tuple[IntVector, FloatMatrix]:
        if self.vectors is None or self._index is None:
            raise RuntimeError("index is not fitted")
        if k < 1:
            raise ValueError("k must be positive")
        query_vectors = _validate_queries(queries, self.vectors.shape[1], self.metric)
        raw_distances, ids = self._index.search(query_vectors, min(k, len(self.vectors)))
        ids = np.asarray(ids, dtype=np.int64)
        distances = np.asarray(raw_distances, dtype=np.float32)
        if self.metric == "cosine":
            distances = 1 - distances
        distances[ids < 0] = np.inf
        return ids, distances

    def memory_bytes(self) -> int:
        if self.vectors is None:
            return 0
        code_bytes = math.ceil(self.subquantizers * self.bits / 8) * len(self.vectors)
        centroid_bytes = self.nlist * self.vectors.shape[1] * 4
        codebook_bytes = self.subquantizers * (1 << self.bits) * (
            self.vectors.shape[1] // self.subquantizers
        ) * 4
        return int(code_bytes + centroid_bytes + codebook_bytes)


class ScaNNIndex(ANNIndex):
    """Optional ScaNN asymmetric-hashing/tree baseline."""

    name = "scann"

    def __init__(self, metric: str = "cosine", leaves: int = 200, leaves_to_search: int = 20):
        super().__init__(metric)
        if leaves < 1 or leaves_to_search < 1:
            raise ValueError("ScaNN leaf parameters must be positive")
        self.leaves = leaves
        self.leaves_to_search = leaves_to_search
        self._index: Any = None

    def fit(self, vectors: Any) -> "ScaNNIndex":
        started = time.perf_counter()
        data = self._set_vectors(vectors)
        try:
            import scann
        except ImportError as error:
            raise RuntimeError("Install ScaNN support with `pip install -e '.[scann]'`.") from error
        count = max(1, min(self.leaves, len(data)))
        self._index = (
            scann.scann_ops_pybind.builder(data, 10, "dot_product" if self.metric == "cosine" else "squared_l2")
            .tree(count, self.leaves_to_search, training_sample_size=len(data))
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build()
        )
        self.build_time_ms = (time.perf_counter() - started) * 1000
        return self

    def search(self, queries: Any, k: int = 10) -> tuple[IntVector, FloatMatrix]:
        if self.vectors is None or self._index is None:
            raise RuntimeError("index is not fitted")
        if k < 1:
            raise ValueError("k must be positive")
        query_vectors = _validate_queries(queries, self.vectors.shape[1], self.metric)
        ids = np.full((len(query_vectors), k), -1, dtype=np.int64)
        distances = np.full((len(query_vectors), k), np.inf, dtype=np.float32)
        for row, query in enumerate(query_vectors):
            result_ids, result_distances = self._index.search(query, final_num_neighbors=min(k, len(self.vectors)))
            count = min(k, len(result_ids))
            ids[row, :count] = result_ids[:count]
            if self.metric == "cosine":
                distances[row, :count] = 1.0 - np.asarray(result_distances[:count], dtype=np.float32)
            else:
                distances[row, :count] = np.asarray(result_distances[:count], dtype=np.float32)
        return ids, distances

    def memory_bytes(self) -> int:
        if self.vectors is None:
            return 0
        return int(self.vectors.nbytes * 1.15)


def make_index(name: str, metric: str = "cosine", **options: Any) -> ANNIndex:
    if name == "learned-rmi":
        return LearnedRMIIndex(metric=metric, **options)
    if name == "ivf-pq":
        return IVFPQIndex(metric=metric, **options)
    if name == "hnsw":
        return HNSWIndex(metric=metric, **options)
    if name == "faiss-ivf-pq":
        return FaissIVFPQIndex(metric=metric, **options)
    if name == "scann":
        return ScaNNIndex(metric=metric, **options)
    raise ValueError("index must be one of: learned-rmi, ivf-pq, hnsw, scann")


def exact_search(
    vectors: Any, queries: Any, k: int = 10, metric: str = "cosine"
) -> tuple[IntVector, FloatMatrix]:
    data = prepare_vectors(vectors, metric)
    query_vectors = _validate_queries(queries, data.shape[1], metric)
    if k < 1:
        raise ValueError("k must be positive")
    ids = np.arange(len(data), dtype=np.int64)
    neighbors = np.full((len(query_vectors), k), -1, dtype=np.int64)
    distances = np.full((len(query_vectors), k), np.inf, dtype=np.float32)
    for row, query in enumerate(query_vectors):
        found_ids, found_distances = _top_k(
            ids,
            _distance(query[None, :], data, metric),
            k,
        )
        neighbors[row, :len(found_ids)] = found_ids
        distances[row, :len(found_distances)] = found_distances
    return neighbors, distances
