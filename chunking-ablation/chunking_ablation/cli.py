"""CLI for the four-strategy chunking benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from chunking_ablation.benchmark import load_documents, load_questions, run_benchmark


def _write_review_csv(result: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "strategy",
                "id",
                "question",
                "expected_answer",
                "evidence",
                "answer",
                "retrieved_contexts",
                "retrieval_recall_at_5",
                "exact_match",
                "llm_judge_correct",
                "human_review",
                "context_tokens",
                "retrieval_latency_ms",
                "end_to_end_latency_ms",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        for run in result["runs"]:
            for row in run["queries"]:
                writer.writerow({"strategy": run["strategy"], **row})


def main() -> None:
    parser = argparse.ArgumentParser(prog="chunking-ablation")
    parser.add_argument("--corpus", type=Path, default=Path("samples/corpus"))
    parser.add_argument("--qa", type=Path, default=Path("samples/qa.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--semantic-threshold", type=float, default=0.12)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--answerer", choices=("extractive", "openai-compatible"), default="extractive")
    parser.add_argument("--judge", action="store_true", help="judge answers with the configured OpenAI-compatible endpoint")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    args = parser.parse_args()
    if (
        args.target_tokens < 1
        or args.overlap_tokens < 0
        or args.overlap_tokens >= args.target_tokens
        or args.window_radius < 0
        or args.input_cost_per_million < 0
        or args.output_cost_per_million < 0
        or not 0 <= args.semantic_threshold <= 1
    ):
        parser.error("invalid chunk size, overlap, window, semantic threshold, or token price")
    try:
        documents = load_documents(args.corpus)
        questions = load_questions(args.qa)
        result = run_benchmark(
            documents,
            questions,
            target_tokens=args.target_tokens,
            overlap_tokens=args.overlap_tokens,
            semantic_threshold=args.semantic_threshold,
            window_radius=args.window_radius,
            llm=args.answerer == "openai-compatible",
            judge=args.judge,
            model=args.model,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "benchmark.json"
        review_path = args.output_dir / "human-review.csv"
        json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_review_csv(result, review_path)
        print(f"Results written to {json_path.resolve()}")
        print(f"Human review sheet written to {review_path.resolve()}")
        print("strategy             recall@5  exact-match  judge  context  latency ms  cost/query")
        for run in result["runs"]:
            judge = "-" if run["llm_judge_accuracy"] is None else f"{run['llm_judge_accuracy']:.3f}"
            print(
                f"{run['strategy']:20} {run['retrieval_recall_at_5']:.3f}    "
                f"{run['exact_answer_accuracy']:.3f}       {judge:>5} "
                f"{run['avg_context_tokens']:7.1f} "
                f"{run['mean_end_to_end_latency_ms']:10.2f} "
                f"${run['avg_cost_per_query']:.7f}"
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"chunking-ablation: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
