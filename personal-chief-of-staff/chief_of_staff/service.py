"""Application service for demo and MCP-connected assistant runs."""

import json
import os
from pathlib import Path
from typing import Any

from chief_of_staff.agents import run_assistant
from chief_of_staff.mcp_config import load_server_config
from chief_of_staff.workflows import run_demo


def demo_enabled() -> bool:
    return os.getenv("CHIEF_OF_STAFF_DEMO", "true").casefold() in {"1", "true", "yes", "on"}


def _read_server_config() -> dict[str, dict[str, Any]]:
    config_path = os.getenv("CHIEF_OF_STAFF_MCP_CONFIG")
    if not config_path:
        return {}
    return load_server_config(Path(config_path))


async def run_task(task: str) -> dict[str, str]:
    if not task.strip():
        raise ValueError("task must not be empty")
    if demo_enabled():
        return run_demo(task)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when CHIEF_OF_STAFF_DEMO is disabled")

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI

    servers = _read_server_config()
    tools = await MultiServerMCPClient(servers).get_tools() if servers else []
    model = ChatOpenAI(
        model=os.getenv("CHIEF_OF_STAFF_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    return await run_assistant(task, model, tools)


def demo_status() -> dict[str, Any]:
    return {
        "mode": "demo" if demo_enabled() else "live",
        "mcp_configured": bool(os.getenv("CHIEF_OF_STAFF_MCP_CONFIG")),
        "writes_enabled": False,
    }
