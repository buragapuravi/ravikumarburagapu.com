"""CLI for the Word2Vec hyperparameter sweep and optional baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from word2vec_lab.baselines import train_fasttext, train_glove
from word2vec_lab.data import load_analogies, load_corpus, load_neighbors
from word2vec_lab.evaluation import evaluate_analogies, evaluate_neighbors
from word2vec_lab.model import SkipGramNS
from word2vec_lab.visualize import plot_analogy_tsne


def _positive_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not values or any(item < 1 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="word2vec-lab")
    parser.add_argument("--corpus", type=Path, default=Path("samples/demo-corpus.txt"))
    parser.add_argument("--analogies", type=Path, default=Path("samples/analogies.jsonl"))
    parser.add_argument("--neighbors", type=Path, default=Path("samples/neighbor-pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dimensions", type=_positive_ints, default=[50, 100, 300])
    parser.add_argument("--windows", type=_positive_ints, default=[2, 5, 10])
    parser.add_argument("--negative-samples", type=_positive_ints, default=[5, 10, 15])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--neighbors-at-k", type=int, default=5)
    parser.add_argument("--baselines", action="store_true", help="also train GloVe and FastText")
    parser.add_argument("--visualize", action="store_true", help="write an analogy t-SNE SVG")
    return parser


def _summary_row(
    name: str,
    dimension: int,
    window: int | None,
    negatives: int | None,
    model: Any,
    analogies: dict[str, Any],
    neighbors: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": name,
        "dimension": dimension,
        "window": window,
        "negative_samples": negatives,
        "vocabulary_size": len(model.words),
        "analogy_total": analogies["total"],
        "analogy_evaluated": analogies["evaluated"],
        "analogy_coverage": analogies["coverage"],
        "analogy_accuracy": analogies["accuracy"],
        "neighbor_total": neighbors["total"],
        "neighbor_evaluated": neighbors["evaluated"],
        "neighbor_precision_at_k": neighbors["precision_at_k"],
        "neighbor_k": neighbors["k"],
        "training_seconds": model.elapsed_seconds,
    }


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.min_count < 1 or args.batch_size < 1 or args.neighbors_at_k < 1:
        parser.error("epochs, min-count, batch-size, and neighbors-at-k must be positive")
    try:
        texts = load_corpus(args.corpus)
        analogies = load_analogies(args.analogies)
        neighbors = load_neighbors(args.neighbors)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        runs: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        visualization_candidates = []
        grid_size = len(args.dimensions) * len(args.windows) * len(args.negative_samples)
        for dimension in args.dimensions:
            for window in args.windows:
                for negative_samples in args.negative_samples:
                    model = SkipGramNS.train(
                        texts,
                        dimension=dimension,
                        window=window,
                        negatives=negative_samples,
                        epochs=args.epochs,
                        min_count=args.min_count,
                        batch_size=args.batch_size,
                        seed=args.seed,
                    )
                    analogy_report = evaluate_analogies(model, analogies)
                    neighbor_report = evaluate_neighbors(
                        model, neighbors, k=args.neighbors_at_k
                    )
                    run_name = f"skipgram-d{dimension}-w{window}-n{negative_samples}"
                    summary = _summary_row(
                        "skipgram",
                        dimension,
                        window,
                        negative_samples,
                        model,
                        analogy_report,
                        neighbor_report,
                    )
                    summaries.append(summary)
                    runs.append(
                        {
                            "name": run_name,
                            "configuration": {
                                "algorithm": "skipgram-negative-sampling",
                                "dimension": dimension,
                                "window": window,
                                "negative_samples": negative_samples,
                                "epochs": args.epochs,
                                "min_count": args.min_count,
                                "batch_size": args.batch_size,
                                "seed": args.seed,
                            },
                            "loss_per_epoch": model.losses,
                            "metrics": summary,
                            "analogies": analogy_report,
                            "neighbors": neighbor_report,
                        }
                    )
                    visualization_candidates.append(
                        (
                            (dimension, window == 5, negative_samples == 5),
                            model,
                        )
                    )
                    print(
                        f"{run_name:26} analogy={analogy_report['accuracy']} "
                        f"coverage={analogy_report['coverage']:.2f} "
                        f"neighbor-P@{args.neighbors_at_k}={neighbor_report['precision_at_k']} "
                        f"train={model.elapsed_seconds:.2f}s"
                    )

        if args.baselines:
            for dimension in args.dimensions:
                baseline_models = (
                    train_glove(
                        texts,
                        dimension=dimension,
                        window=5,
                        epochs=args.epochs,
                        min_count=args.min_count,
                        seed=args.seed,
                    ),
                    train_fasttext(
                        texts,
                        dimension=dimension,
                        window=5,
                        epochs=args.epochs,
                        min_count=args.min_count,
                        seed=args.seed,
                    ),
                )
                for model in baseline_models:
                    analogy_report = evaluate_analogies(model, analogies)
                    neighbor_report = evaluate_neighbors(
                        model, neighbors, k=args.neighbors_at_k
                    )
                    summary = _summary_row(
                        model.algorithm,
                        dimension,
                        5,
                        None,
                        model,
                        analogy_report,
                        neighbor_report,
                    )
                    summaries.append(summary)
                    runs.append(
                        {
                            "name": f"{model.algorithm}-d{dimension}-w5",
                            "configuration": {
                                "algorithm": model.algorithm,
                                "dimension": dimension,
                                "window": 5,
                                "epochs": args.epochs,
                                "min_count": args.min_count,
                                "seed": args.seed,
                            },
                            "metrics": summary,
                            "analogies": analogy_report,
                            "neighbors": neighbor_report,
                        }
                    )
                    print(
                        f"{model.algorithm:26} d={dimension:<3} "
                        f"analogy={analogy_report['accuracy']} "
                        f"neighbor-P@{args.neighbors_at_k}={neighbor_report['precision_at_k']} "
                        f"train={model.elapsed_seconds:.2f}s"
                    )

        report = {
            "experiment": "skip-gram negative sampling hyperparameter sweep",
            "corpus": str(args.corpus),
            "corpus_lines": len(texts),
            "analogy_fixture": str(args.analogies),
            "neighbor_fixture": str(args.neighbors),
            "fixture_notice": (
                "Bundled text and labels are original illustrative examples, not an "
                "externally validated benchmark or the user's private corpus."
            ),
            "sweep": {
                "dimensions": args.dimensions,
                "windows": args.windows,
                "negative_samples": args.negative_samples,
                "skipgram_configurations": grid_size,
                "epochs": args.epochs,
                "min_count": args.min_count,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            "baselines_included": args.baselines,
            "runs": runs,
        }
        json_path = args.output_dir / "report.json"
        csv_path = args.output_dir / "summary.csv"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _save_csv(summaries, csv_path)
        if args.visualize:
            _, selected_model = max(visualization_candidates, key=lambda item: item[0])
            svg_path = args.output_dir / "analogy-tsne.svg"
            plot_analogy_tsne(selected_model, analogies, svg_path, seed=args.seed)
            print(f"t-SNE visualization written to {svg_path.resolve()}")
        print(f"Full report written to {json_path.resolve()}")
        print(f"Sweep summary written to {csv_path.resolve()}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"word2vec-lab: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
