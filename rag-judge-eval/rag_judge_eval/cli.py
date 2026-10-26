"""Command-line interface for preparing labels and running judge comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag_judge_eval.harness import (
    load_annotations,
    load_cases,
    run_evaluation,
    write_annotation_template,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-judge-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write a human annotation CSV template")
    prepare.add_argument("--cases", type=Path, default=Path("samples/cases.jsonl"))
    prepare.add_argument("--review", type=Path, default=Path("results/human-review.csv"))
    run = commands.add_parser("run", help="run three judges and calculate agreement")
    run.add_argument("--cases", type=Path, default=Path("samples/cases.jsonl"))
    run.add_argument("--annotations", type=Path, default=Path("results/human-review.csv"))
    run.add_argument("--output", type=Path, default=Path("results/report.json"))
    run.add_argument("--model", default="gpt-4o-mini")
    run.add_argument("--input-cost-per-million", type=float, default=0.0)
    run.add_argument("--output-cost-per-million", type=float, default=0.0)
    run.add_argument("--human-hourly-rate", type=float)
    run.add_argument("--human-minutes-per-case", type=float)
    run.add_argument("--timeout", type=float, default=90)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "run":
        if (
            args.input_cost_per_million < 0
            or args.output_cost_per_million < 0
            or args.timeout <= 0
            or (args.human_hourly_rate is not None and args.human_hourly_rate < 0)
            or (args.human_minutes_per_case is not None and args.human_minutes_per_case < 0)
            or ((args.human_hourly_rate is None) != (args.human_minutes_per_case is None))
        ):
            parser.error("prices and time must be non-negative; provide both human-cost options")
    try:
        cases = load_cases(args.cases)
        if args.command == "prepare":
            write_annotation_template(cases, args.review)
            print(f"Annotation template written to {args.review.resolve()}")
            print("Labels are blank; complete them independently before reviewing judge results.")
            return
        annotations = load_annotations(args.annotations, cases)
        report = run_evaluation(
            cases,
            annotations,
            model=args.model,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
            human_hourly_rate=args.human_hourly_rate,
            human_minutes_per_case=args.human_minutes_per_case,
            timeout=args.timeout,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.output.resolve()}")
        print("prompt             n  answer agreement  Cohen's kappa  citation agreement  cost/1K cases")
        for name, result in report["prompts"].items():
            kappa = result["answer_kappa"]
            kappa_text = "n/a" if kappa is None else f"{kappa:.3f}"
            agreement = result["answer_agreement"]
            agreement_text = "n/a" if agreement is None else f"{agreement:.3f}"
            citation = result["citation_agreement"]
            citation_text = "n/a" if citation is None else f"{citation:.3f}"
            print(
                f"{name:19} {result['n_labeled']:2} {agreement_text:>16} "
                f"{kappa_text:>14} {citation_text:>20} "
                f"${result['cost_per_1000_cases_usd']:.4f}"
            )
        if report["human_label_cost"] is not None:
            print(
                "Human label cost/1K: "
                f"${report['human_label_cost']['cost_per_1000_labels_usd']:.2f}"
            )
        if report["human_labeled_cases"] < report["total_cases"]:
            print(
                f"Agreement uses {report['human_labeled_cases']} fully labeled pairs of "
                f"{report['total_cases']}; unlabeled rows are excluded."
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"rag-judge-eval: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
