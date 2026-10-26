"""Train matched parameter language models and long-range recall probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from sequence_models.experiment import (
    encode_corpus,
    make_recall_data,
    train_language_model,
    train_recall_model,
)
from sequence_models.models import MODEL_NAMES, matched_configs


def _svg_chart(runs: list[dict[str, Any]]) -> str:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 40, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    losses = [
        point["validation_loss"]
        for run in runs
        for point in run["history"]
        if "validation_loss" in point
    ]
    low = min(losses) if losses else 0.0
    high = max(losses) if losses else 1.0
    span = max(0.01, high - low)
    palette = ("#0077b6", "#d62828", "#2a9d8f", "#8338ec")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Validation loss vs training steps</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Validation cross-entropy</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">Training steps</text>',
    ]
    max_step = max(
        (point["step"] for run in runs for point in run["history"] if "validation_loss" in point),
        default=1,
    )
    for index, run in enumerate(runs):
        points = [
            point for point in run["history"] if "validation_loss" in point
        ]
        coordinates = []
        for point in points:
            x = margin_left + plot_width * point["step"] / max_step
            y = margin_top + plot_height * (high - point["validation_loss"]) / span
            coordinates.append(f"{x:.1f},{y:.1f}")
        color = palette[index % len(palette)]
        if coordinates:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(coordinates)}"/>'
            )
        legend_y = margin_top + index * 24
        parts.append(f'<line x1="{width - 220}" y1="{legend_y}" x2="{width - 190}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - 182}" y="{legend_y + 5}" font-family="sans-serif" font-size="13">{run["model"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    train_data, valid_data, alphabet = encode_corpus(args.corpus, args.corpus_repeats)
    configs = matched_configs(
        len(alphabet),
        target_parameters=args.target_parameters,
        recurrent_layers=1,
        transformer_layers=2,
        transformer_heads=4,
        max_sequence_length=args.sequence_length + 1,
    )
    language_runs = []
    samples = []
    for name in MODEL_NAMES:
        model, metrics, sample = train_language_model(
            configs[name],
            train_data,
            valid_data,
            alphabet,
            target_parameters=args.target_parameters,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=device,
            seed=args.seed,
        )
        language_runs.append(metrics)
        samples.append({"model": name, "characters": sample})
        del model

    recall_train_x, recall_train_y = make_recall_data(args.recall_train, args.recall_length, args.seed + 1)
    recall_valid_x, recall_valid_y = make_recall_data(args.recall_valid, args.recall_length, args.seed + 2)
    recall_runs = [
        train_recall_model(
            configs[name],
            recall_train_x,
            recall_train_y,
            recall_valid_x,
            recall_valid_y,
            epochs=args.recall_epochs,
            batch_size=args.recall_batch_size,
            device=device,
            seed=args.seed,
        )
        for name in MODEL_NAMES
    ]
    return {
        "corpus": {
            "path": str(args.corpus),
            "training_characters": len(train_data),
            "validation_characters": len(valid_data),
            "vocabulary_size": len(alphabet),
            "repeats": args.corpus_repeats,
        },
        "language_modeling": {
            "runs": language_runs,
            "samples": samples,
            "training": {
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "optimizer": "AdamW",
                "learning_rate": 3e-4,
                "device": str(device),
                "threads": args.threads,
            },
        },
        "long_range_recall": {
            "description": "The final query marker requires recalling the binary token at position 0 across distractors.",
            "runs": recall_runs,
            "training": {
                "sequence_length": args.recall_length,
                "train_examples": args.recall_train,
                "validation_examples": args.recall_valid,
                "epochs": args.recall_epochs,
                "random_baseline": 0.5,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="sequence-models")
    parser.add_argument("--corpus", type=Path, default=Path("samples/corpus.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-parameters", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--corpus-repeats", type=int, default=20)
    parser.add_argument("--recall-length", type=int, default=101)
    parser.add_argument("--recall-train", type=int, default=512)
    parser.add_argument("--recall-valid", type=int, default=128)
    parser.add_argument("--recall-epochs", type=int, default=2)
    parser.add_argument("--recall-batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.target_parameters < 10_000 or min(
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.sequence_length,
        args.corpus_repeats,
        args.recall_length,
        args.recall_train,
        args.recall_valid,
        args.recall_epochs,
        args.recall_batch_size,
        args.threads,
    ) < 1:
        parser.error("model size, training lengths, budgets, and thread count must be positive")
    try:
        result = run(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "samples.txt").write_text(
            "\n\n".join(f"## {item['model']}\n{item['characters']}" for item in result["language_modeling"]["samples"]) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "validation-loss.svg").write_text(
            _svg_chart(result["language_modeling"]["runs"]), encoding="utf-8"
        )
        print(f"Results written to {args.output_dir.resolve()}")
        for language, recall in zip(
            result["language_modeling"]["runs"],
            result["long_range_recall"]["runs"],
            strict=True,
        ):
            print(
                f"{language['model']:12} params={language['parameters']:,} "
                f"ppl={language['perplexity']:.2f} "
                f"recall@100={recall['accuracy']:.3f}"
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"sequence-models: {error}\n")


if __name__ == "__main__":
    main()
