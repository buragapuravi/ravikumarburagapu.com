"""Load trusted MCP server definitions with environment-only secret references."""

import json
import os
import re
from pathlib import Path
from typing import Any

_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CLIENT_KEY")


def load_server_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Validate stdio server config and resolve ${ENV_VAR} values without logging them."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{config_path}: invalid JSON: {error.msg}") from error
    if not isinstance(raw, dict) or not raw:
        raise ValueError("MCP config must be a non-empty object keyed by server name")

    resolved: dict[str, dict[str, Any]] = {}
    for name, server in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(server, dict):
            raise ValueError("each MCP server must have a non-empty name and object config")
        if server.get("transport", "stdio") != "stdio":
            raise ValueError(f"MCP server '{name}': only stdio transport is supported")
        command = server.get("command")
        args = server.get("args", [])
        if not isinstance(command, str) or not command.strip() or not isinstance(args, list):
            raise ValueError(f"MCP server '{name}': command must be a string and args must be a list")
        if not all(isinstance(argument, str) for argument in args):
            raise ValueError(f"MCP server '{name}': args must contain only strings")
        env = server.get("env", {})
        if not isinstance(env, dict):
            raise ValueError(f"MCP server '{name}': env must be an object")
        safe_env: dict[str, str] = {}
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f"MCP server '{name}': environment names and values must be strings")
            reference = _ENV_REFERENCE.fullmatch(value)
            if reference:
                variable = reference.group(1)
                if variable not in os.environ:
                    raise ValueError(f"MCP server '{name}': required environment variable {variable} is not set")
                safe_env[key] = os.environ[variable]
            elif any(secret_word in key.upper() for secret_word in _SECRET_WORDS):
                raise ValueError(
                    f"MCP server '{name}': secret environment values must use a ${{ENV_VAR}} reference"
                )
            else:
                safe_env[key] = value
        resolved[name] = {
            "transport": "stdio",
            "command": command,
            "args": args,
            "env": safe_env,
        }
    return resolved
