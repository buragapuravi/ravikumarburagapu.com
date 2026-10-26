"""Command-line interface for the Personal Chief of Staff."""

import argparse
import asyncio
import json
import sys

import uvicorn

from chief_of_staff.service import run_task


async def _run(task: str, output: str) -> None:
    result = await run_task(task)
    if output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['mode']} · {result['workflow']}]\n\n{result['response']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="chief-of-staff")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run", help="run one assistant workflow")
    run_parser.add_argument("task", nargs="+")
    run_parser.add_argument("--format", choices=("text", "json"), default="text")
    web_parser = subcommands.add_parser("web", help="start the local web UI")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    try:
        if args.command == "run":
            asyncio.run(_run(" ".join(args.task), args.format))
        else:
            if args.host not in {"127.0.0.1", "localhost", "::1"}:
                parser.error("the web UI binds to localhost only")
            uvicorn.run("chief_of_staff.app:app", host=args.host, port=args.port, reload=False)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"chief-of-staff: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
