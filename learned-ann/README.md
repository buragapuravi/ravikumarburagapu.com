# Learned ANN Index

An experimental recursive model index (RMI) for approximate nearest-neighbor search. A hierarchy of learned k-means routing decisions selects candidate partitions, followed by exact scan or optional local HNSW within selected leaves. Compare recall@10, QPS, index build time, and approximate index memory with residual IVF-PQ, plus optional Faiss IVF-PQ, HNSW, and ScaNN adapters.

> **Research prototype:** This repository demonstrates a learned partitioning design. Its synthetic benchmark is not evidence that a learned partitioner outperforms mature ANN systems on production embeddings. Report hardware, dataset, metric, parameters, and repeat trials before making comparisons.

## Quick start

Requires Python 3.11+.

```sh
cd learned-ann
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"

# Self-contained deterministic smoke benchmark; no external data required.
learned-ann benchmark \
  --vectors 4000 --queries 160 --dimension 32 \
  --indexes learned-rmi ivf-pq \
  --output results/ann-benchmark.json

# Optional interactive recall/QPS plot.
streamlit run dashboard.py
```

The built-in algorithms use NumPy only. Optional native baselines:

```sh
pip install -e ".[hnsw]"       # hnswlib
pip install -e ".[faiss]"      # faiss-cpu, native Faiss IVF-PQ baseline
pip install -e ".[scann]"      # ScaNN
pip install -e ".[dashboard]"  # Streamlit dashboard and pandas

learned-ann benchmark --indexes learned-rmi ivf-pq faiss-ivf-pq hnsw scann
```

An unavailable optional backend is reported in JSON rather than silently replaced.

## Algorithms

### Learned recursive model index

The experimental **RMI-style partitioner** recursively fits k-means routing nodes until leaf partitions satisfy `max_leaf_size`. At query time, it descends the hierarchy while retaining `nprobe` nearest child centers and searches the selected leaves. The default hybrid scans each leaf exactly; `--inner-backend hnsw` optionally uses local hnswlib indexes inside leaves. `--branch-factor`, `--max-leaf-size`, and `--nprobe` control the learned hierarchy and candidate budget.

This is a learned hierarchical clustering/routing model—not a learned guarantee of nearest-neighbor rank or an exact extension of the original sorted-key RMI. Its centroid routing can miss close points in unprobed leaves. Measure recall for every workload.

### IVF-PQ

The NumPy baseline trains coarse inverted lists, encodes residuals using product-quantization codebooks, probes nearest coarse centers, and exactly reranks a short approximate-code shortlist. Set `--nlist`, `--nprobe`, `--subquantizers`, and `--bits`. Embedding dimension must be divisible by the number of subquantizers. For a standardized and optimized native baseline, install `.[faiss]` and select `faiss-ivf-pq`; the default `ivf-pq` is a compact NumPy residual-PQ implementation.

### Optional HNSW and ScaNN

`hnswlib` provides a conventional HNSW baseline. The optional `faiss-ivf-pq` adapter provides native Faiss IVF-PQ. The optional ScaNN adapter uses tree partitioning, asymmetric hashing, and exact reorder on its candidate set. Native packages may have platform, Python-version, and hardware constraints; installation is not required for the default demo.

## Benchmark and evaluation

The command generates clustered, normalized synthetic cosine embeddings with a fixed seed, uses brute-force exact search as ground truth, and reports:

- **recall@k:** fraction of exact top-k IDs present in each approximate top-k result.
- **QPS / mean query latency:** sequential query-loop throughput in this Python process; not a concurrent service capacity test.
- **Build time:** wall-clock fit time for the selected implementation.
- **Memory estimate:** tracked arrays for NumPy indexes and analytical data/code estimates for Faiss; HNSW uses a rough graph-link allowance and ScaNN uses a vector-size multiplier. These estimates are not process RSS and exclude allocator/runtime overhead.

The dashboard marks non-dominated runs on the recall/QPS plot and displays their recall frontier. A point is non-dominated when no other run has both at least its recall and at least its QPS, with a strict improvement on one axis.

Use your own `.npy` vector matrix or `.npz` containing `vectors` and `queries`:

```sh
learned-ann benchmark \
  --input /path/to/authorized-embeddings.npz \
  --metric cosine \
  --indexes learned-rmi ivf-pq hnsw \
  --output results/my-run.json
```

Inputs are loaded with `allow_pickle=False`. For valid experiments, normalize and deduplicate the dataset consistently, use representative held-out queries, run repeated trials, and publish all index parameters. QPS depends heavily on Python, BLAS, thread settings, CPU/GPU, and batch size. The current harness executes query vectors sequentially and does not claim statistical confidence intervals.

## Development

```sh
pip install -e ".[dev]"
pytest
```

Please add tests and state dataset, normalization, hardware, dependency version, accuracy, and memory assumptions for benchmark contributions.

## License

MIT. See [LICENSE](LICENSE).
