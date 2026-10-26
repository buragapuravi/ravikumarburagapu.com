"""Command-line entry point for inference and RAG benchmark runs."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from benchmarkkit.datasets import load_jsonl, prepare_public_questions, validate_prompts
from benchmarkkit.inference import run_inference
from benchmarkkit.metrics import summarize_rag


def _write_result(result: dict[str, Any], output: str) -> None:
    serialized = json.dumps(result, indent=2)
    if output == "-":
        print(serialized)
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized + "\n", encoding="utf-8")
    print(f"Results written to {destination}")


def _inference_command(args: argparse.Namespace) -> None:
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    source_prompts = validate_prompts(load_jsonl(args.dataset))
    prompts = [source_prompts[index % len(source_prompts)] for index in range(args.requests)]
    if not prompts:
        raise ValueError("the dataset has no prompts to run")
    result = asyncio.run(
        run_inference(
            prompts=prompts,
            endpoint=args.endpoint,
            model=args.model,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            input_price=args.input_price,
            output_price=args.output_price,
            streaming=not args.no_stream,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            timeout_seconds=args.timeout,
            metrics_port=args.metrics_port,
            framework=args.framework,
        )
    )
    result["framework"] = args.framework
    result["model"] = args.model
    result["endpoint"] = args.endpoint
    _write_result(result, args.output)


def _rag_command(args: argparse.Namespace) -> None:
    result = summarize_rag(load_jsonl(args.dataset))
    _write_result(result, args.output)


def _prepare_command(args: argparse.Namespace) -> None:
    records = prepare_public_questions(args.input, args.dataset, args.limit)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Prepared {len(records)} prompts in {destination}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarkkit", description="Benchmark LLM inference and RAG pipelines.")
    commands = parser.add_subparsers(dest="command", required=True)

    inference = commands.add_parser("inference", help="load-test an OpenAI-compatible chat completions endpoint")
    inference.add_argument("--dataset", required=True, help="JSONL file with one {\"prompt\": \"...\"} per line")
    inference.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    inference.add_argument("--model", required=True)
    inference.add_argument("--framework", default="openai-compatible", help="label used in results and Prometheus")
    inference.add_argument("--requests", type=int, default=100)
    inference.add_argument("--concurrency", type=int, default=8)
    inference.add_argument("--max-tokens", type=int, default=128)
    inference.add_argument("--input-price", type=float, default=0, help="USD per 1M input tokens")
    inference.add_argument("--output-price", type=float, default=0, help="USD per 1M output tokens")
    inference.add_argument("--api-key", help="API key; defaults to OPENAI_API_KEY")
    inference.add_argument("--timeout", type=float, default=120)
    inference.add_argument("--metrics-port", type=int, default=9464, help="Prometheus exporter port; 0 disables it")
    inference.add_argument("--no-stream", action="store_true", help="use non-streaming completions")
    inference.add_argument("--output", default="results/inference.json", help="output JSON path or '-' for stdout")
    inference.set_defaults(handler=_inference_command)

    rag = commands.add_parser("rag", help="evaluate retrieval and answer records saved as JSONL")
    rag.add_argument("--dataset", required=True)
    rag.add_argument("--output", default="results/rag.json", help="output JSON path or '-' for stdout")
    rag.set_defaults(handler=_rag_command)

    prepare = commands.add_parser("prepare", help="convert public QA datasets to prompt JSONL")
    prepare.add_argument("--dataset", required=True, choices=("hotpotqa", "natural-questions"))
    prepare.add_argument("--input", required=True, help="downloaded source dataset file")
    prepare.add_argument("--limit", type=int, default=100, help="maximum questions to prepare")
    prepare.add_argument("--output", default="samples/public-prompts.jsonl")
    prepare.set_defaults(handler=_prepare_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, httpx.HTTPError) as error:
        print(f"benchmarkkit: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
