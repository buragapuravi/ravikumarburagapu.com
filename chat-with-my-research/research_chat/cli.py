"""CLI for corpus ingestion, cited queries, and retrieval evaluation."""

import argparse
import json
import sys
from pathlib import Path

from research_chat.corpus import DEFAULT_DB, ingest_directory
from research_chat.decomposition import (
    ALGORITHMS,
    DecompositionSearch,
    evaluate_algorithms,
    make_live_components,
)
from research_chat.evaluation import evaluate_example, summarize_evaluations
from research_chat.rag import create_rag_graph, run_query


def _ingest(args: argparse.Namespace) -> None:
    print(json.dumps(ingest_directory(args.directory, args.db, args.chunk_chars), indent=2))


def _ask(args: argparse.Namespace) -> None:
    result = run_query(args.question, args.db, args.top_k)
    print(result["answer"])
    print(f"\nMode: {result['mode']} · Retrieval/reranking: {result['reranker']}")
    for source in result["sources"]:
        print(f"[{source['citation']}] {source['title']} — {source['source_path']} · {source['section']}")


def _evaluate(args: argparse.Namespace) -> None:
    examples = []
    with Path(args.dataset).open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.dataset}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(example, dict):
                raise ValueError(f"{args.dataset}:{line_number}: expected a JSON object")
            examples.append(example)
    if not examples:
        raise ValueError("evaluation dataset contains no examples")
    graph = create_rag_graph(args.db, args.top_k)
    results = [
        evaluate_example(graph, example, args.recall_k)
        for example in examples
    ]
    summary = summarize_evaluations(results)
    serialized = json.dumps(summary, indent=2)
    if args.output == "-":
        print(serialized)
    else:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
        print(f"Evaluation written to {destination}")


def _search(args: argparse.Namespace) -> None:
    proposer, value_function = make_live_components(args.model)
    result = DecompositionSearch(
        args.db,
        top_k=args.top_k,
        proposer=proposer,
        value_function=value_function,
    ).run(
        args.question,
        algorithm=args.algorithm,
        max_hops=args.max_hops,
        iterations=args.iterations,
        beam_width=args.beam_width,
        alpha=args.alpha,
        call_cost=args.call_cost,
    )
    print(json.dumps(result, indent=2))


def _compare(args: argparse.Namespace) -> None:
    examples = []
    with Path(args.dataset).open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.dataset}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(example, dict):
                raise ValueError(f"{args.dataset}:{line_number}: expected a JSON object")
            examples.append(example)
    proposer, value_function = make_live_components(args.model)
    result = evaluate_algorithms(
        examples,
        algorithms=args.algorithms or list(ALGORITHMS),
        db_path=args.db,
        top_k=args.top_k,
        proposer=proposer,
        value_function=value_function,
        max_hops=args.max_hops,
        iterations=args.iterations,
        beam_width=args.beam_width,
        alpha=args.alpha,
        call_cost=args.call_cost,
        recall_k=args.recall_k,
    )
    serialized = json.dumps(result, indent=2)
    if args.output == "-":
        print(serialized)
    else:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
        print(f"Planner comparison written to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="research-chat")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index Markdown and text documents")
    ingest.add_argument("directory")
    ingest.add_argument("--db", default=str(DEFAULT_DB))
    ingest.add_argument("--chunk-chars", type=int, default=1400)
    ingest.set_defaults(handler=_ingest)

    ask = commands.add_parser("ask", help="ask a question and print source citations")
    ask.add_argument("question")
    ask.add_argument("--db", default=str(DEFAULT_DB))
    ask.add_argument("--top-k", type=int, default=5)
    ask.set_defaults(handler=_ask)

    evaluate = commands.add_parser("evaluate", help="run retrieval and citation metrics on labeled JSONL")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--db", default=str(DEFAULT_DB))
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--recall-k", type=int, default=5)
    evaluate.add_argument("--output", default="-")
    evaluate.set_defaults(handler=_evaluate)

    search = commands.add_parser("search", help="search decomposition trees for a multi-hop question")
    search.add_argument("question")
    search.add_argument("--db", default=str(DEFAULT_DB))
    search.add_argument("--algorithm", choices=ALGORITHMS, default="mcts")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--max-hops", type=int, default=3)
    search.add_argument("--iterations", type=int, default=20)
    search.add_argument("--beam-width", type=int, default=4)
    search.add_argument("--alpha", type=float, default=0.05)
    search.add_argument("--call-cost", type=float, default=0.03)
    search.add_argument("--model", help="override RESEARCH_CHAT_MODEL")
    search.set_defaults(handler=_search)

    compare = commands.add_parser("compare", help="compare decomposition strategies on labeled JSONL")
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--db", default=str(DEFAULT_DB))
    compare.add_argument("--algorithms", nargs="+", choices=ALGORITHMS)
    compare.add_argument("--top-k", type=int, default=5)
    compare.add_argument("--max-hops", type=int, default=3)
    compare.add_argument("--iterations", type=int, default=20)
    compare.add_argument("--beam-width", type=int, default=4)
    compare.add_argument("--alpha", type=float, default=0.05)
    compare.add_argument("--call-cost", type=float, default=0.03)
    compare.add_argument("--recall-k", type=int, default=5)
    compare.add_argument("--model", help="override RESEARCH_CHAT_MODEL")
    compare.add_argument("--output", default="-")
    compare.set_defaults(handler=_compare)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"research-chat: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
