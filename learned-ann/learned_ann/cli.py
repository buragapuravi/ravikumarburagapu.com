"""Command line interface for the learned ANN experiments."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from learned_ann.benchmark import benchmark_index, synthetic_data


def _read_vectors(path: str, query_count: int) -> tuple[Any, Any]:
    import numpy as np

    file_path = Path(path)
    if file_path.suffix.casefold() == ".npz":
        with np.load(file_path, allow_pickle=False) as data:
            if "vectors" not in data or "queries" not in data:
                raise ValueError("NPZ input must contain 'vectors' and 'queries' arrays")
            return data["vectors"], data["queries"]
    vectors = np.load(file_path, allow_pickle=False)
    if vectors.ndim != 2:
        raise ValueError("input NPY vectors must be a 2D array")
    if len(vectors) < 2:
        raise ValueError("input NPY must contain at least two vectors")
    split = min(max(1, len(vectors) - query_count), int(len(vectors) * 0.8))
    if split >= len(vectors):
        split = len(vectors) - 1
    return vectors[:split], vectors[split:]


def _options(args: argparse.Namespace, index_name: str) -> dict[str, Any]:
    if index_name == "learned-rmi":
        return {
            "branch_factor": args.branch_factor,
            "max_leaf_size": args.max_leaf_size,
            "nprobe": args.nprobe,
            "inner_backend": args.inner_backend,
        }
    if index_name in {"ivf-pq", "faiss-ivf-pq"}:
        return {
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "subquantizers": args.subquantizers,
            "bits": args.bits,
        }
    if index_name == "hnsw":
        return {"m": args.hnsw_m, "ef_search": args.ef_search}
    if index_name == "scann":
        return {"leaves": args.nlist, "leaves_to_search": args.nprobe}
    return {}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        vectors, queries = _read_vectors(args.input, args.queries)
    else:
        vectors, queries, _ = synthetic_data(
            vector_count=args.vectors,
            query_count=args.queries,
            dimension=args.dimension,
            cluster_count=args.clusters,
            seed=args.seed,
        )
    results = []
    errors = []
    for name in args.indexes:
        try:
            results.append(
                benchmark_index(
                    name,
                    vectors,
                    queries,
                    k=args.k,
                    metric=args.metric,
                    index_options=_options(args, name),
                )
            )
        except RuntimeError as error:
            errors.append({"index": name, "error": str(error)})
    return {
        "dataset": {
            "kind": "input" if args.input else "synthetic",
            "vector_count": len(vectors),
            "query_count": len(queries),
            "dimension": vectors.shape[1],
            "seed": None if args.input else args.seed,
        },
        "results": results,
        "unavailable": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learned-ann")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="compare ANN indexes on vectors or generated data")
    benchmark.add_argument("--input", help="NPY vectors or NPZ containing vectors and queries")
    benchmark.add_argument(
        "--indexes",
        nargs="+",
        choices=("learned-rmi", "ivf-pq", "faiss-ivf-pq", "hnsw", "scann"),
        default=["learned-rmi", "ivf-pq"],
    )
    benchmark.add_argument("--vectors", type=int, default=4000)
    benchmark.add_argument("--queries", type=int, default=160)
    benchmark.add_argument("--dimension", type=int, default=32)
    benchmark.add_argument("--clusters", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=41)
    benchmark.add_argument("--k", type=int, default=10)
    benchmark.add_argument("--metric", choices=("cosine", "l2"), default="cosine")
    benchmark.add_argument("--branch-factor", type=int, default=4)
    benchmark.add_argument("--max-leaf-size", type=int, default=128)
    benchmark.add_argument("--nlist", type=int, default=32)
    benchmark.add_argument("--nprobe", type=int, default=4)
    benchmark.add_argument("--subquantizers", type=int, default=8)
    benchmark.add_argument("--bits", type=int, default=8)
    benchmark.add_argument("--inner-backend", choices=("exact", "hnsw"), default="exact")
    benchmark.add_argument("--hnsw-m", type=int, default=16)
    benchmark.add_argument("--ef-search", type=int, default=64)
    benchmark.add_argument("--output", default="-")
    benchmark.set_defaults(handler=_run)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = args.handler(args)
        output = json.dumps(result, indent=2)
        if args.output == "-":
            print(output)
        else:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(output + "\n", encoding="utf-8")
            print(f"Benchmark results written to {destination}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"learned-ann: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
