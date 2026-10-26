"""Run tokenizer size and reference-tokenizer downstream comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from tokenizer_ablation.bpe import train_bpe
from tokenizer_ablation.experiment import (
    LABEL_NAMES,
    evaluate,
    make_dataset,
    make_tokenizer_corpus,
    tokenizer_metrics,
)


def _references() -> tuple[dict[str, Any], list[dict[str, str]]]:
    references: dict[str, Any] = {}
    unavailable = []
    try:
        import tiktoken

        references["tiktoken-cl100k"] = tiktoken.get_encoding("cl100k_base")
    except ImportError as error:
        unavailable.append({"tokenizer": "tiktoken-cl100k", "error": f"install .[tiktoken]: {error}"})
    try:
        from transformers import AutoTokenizer

        references["transformers-wordpiece"] = AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True
        )
    except (ImportError, OSError, ValueError) as error:
        unavailable.append({"tokenizer": "transformers-wordpiece", "error": str(error)})
    return references, unavailable


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_data, test_data = make_dataset(seed=args.seed)
    supervised_identifiers = {
        example.identifier for example in train_data + test_data
    }
    training_texts = make_tokenizer_corpus(
        supervised_identifiers,
        count=args.tokenizer_corpus_limit,
        seed=args.seed + 1000,
    )
    started = perf_counter()
    merges = train_bpe(training_texts, max(args.vocab_sizes))
    bpe_training_seconds = perf_counter() - started
    tokenizers: list[tuple[str, Any, int | None]] = [
        (
            f"scratch-bpe-{size}",
            type(merges)(merges.merges, min(size, merges.vocab_size)),
            min(size, merges.vocab_size),
        )
        for size in args.vocab_sizes
    ]
    unavailable: list[dict[str, str]] = []
    if args.with_references:
        references, unavailable = _references()
        tokenizers.extend((name, tokenizer, None) for name, tokenizer in references.items())

    test_metrics = {}
    runs = []
    for name, tokenizer, actual_vocab_size in tokenizers:
        test_metrics[name] = tokenizer_metrics(tokenizer, test_data)
        for budget_kind, budget in (("tokens", args.token_budget), ("bytes", args.byte_budget)):
            result = evaluate(
                tokenizer,
                train_data,
                test_data,
                budget_kind=budget_kind,
                budget=budget,
                feature_width=args.feature_width,
            )
            runs.append(
                {
                    "tokenizer": name,
                    "vocab_size": actual_vocab_size,
                    "tokenizer_training_seconds": bpe_training_seconds if actual_vocab_size else None,
                    **result,
                }
            )
    return {
        "dataset": {
            "name": "generated-code-identifiers",
            "train_examples": len(train_data),
            "test_examples": len(test_data),
            "labels": 8,
            "label_names": LABEL_NAMES,
            "seed": args.seed,
            "training_corpus": "separate unlabeled synthetic identifier corpus; no classifier identifiers included",
            "tokenizer_training_examples": len(training_texts),
            "tokenizer_training_seconds": bpe_training_seconds,
        },
        "tokenizer_metrics_on_test": test_metrics,
        "runs": runs,
        "unavailable_references": unavailable,
        "method": {
            "model": "fixed-width hashed bag-of-subword multinomial logistic classifier",
            "feature_width": args.feature_width,
            "epochs": 6,
            "fixed_token_budget": args.token_budget,
            "fixed_byte_budget": args.byte_budget,
            "notes": [
                "The same classifier architecture and optimizer are used for every tokenizer.",
                "The tokenizer vocabulary size changes feature hashing collisions and identifier segmentation.",
                "Training time excludes tokenizer training and encoding; it is the measured classifier fit time.",
                "Reference tokenizers may require network access to download their model files.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="tokenizer-ablation")
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[1000, 4000, 16000, 32000])
    parser.add_argument("--token-budget", type=int, default=10_000)
    parser.add_argument("--byte-budget", type=int, default=50_000)
    parser.add_argument("--feature-width", type=int, default=2048)
    parser.add_argument("--tokenizer-corpus-limit", type=int, default=35_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--with-references", action="store_true")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if any(size < 256 for size in args.vocab_sizes):
        parser.error("vocab sizes must be at least 256 for byte fallback")
    if (
        args.token_budget < 1
        or args.byte_budget < 1
        or args.feature_width < 1
        or args.tokenizer_corpus_limit < 1
    ):
        parser.error("budgets, feature width, and tokenizer corpus limit must be positive")
    try:
        result = run(args)
        output = json.dumps(result, indent=2)
        if args.output == "-":
            print(output)
        else:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output + "\n", encoding="utf-8")
            print(f"Results written to {path}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"tokenizer-ablation: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
