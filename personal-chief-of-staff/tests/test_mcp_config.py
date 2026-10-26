import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.tools import tool

from chief_of_staff.agents import is_read_only_tool
from chief_of_staff.mcp_config import load_server_config


class MCPConfigTests(unittest.TestCase):
    def test_resolves_secret_from_environment(self) -> None:
        config = {
            "calendar": {
                "transport": "stdio",
                "command": "trusted-server",
                "args": ["--stdio"],
                "env": {"CALENDAR_ACCESS_TOKEN": "${CALENDAR_ACCESS_TOKEN}"},
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "servers.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(os.environ, {"CALENDAR_ACCESS_TOKEN": "test-secret"}):
                resolved = load_server_config(path)

        self.assertEqual(resolved["calendar"]["env"]["CALENDAR_ACCESS_TOKEN"], "test-secret")
        self.assertEqual(resolved["calendar"]["transport"], "stdio")

    def test_rejects_inline_secret(self) -> None:
        config = {
            "calendar": {
                "command": "trusted-server",
                "env": {"CALENDAR_ACCESS_TOKEN": "do-not-inline"},
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "servers.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must use"):
                load_server_config(path)


class ToolPermissionTests(unittest.TestCase):
    def test_tool_filter_fails_closed_for_writes_and_unknown_names(self) -> None:
        @tool
        def list_calendar_events() -> str:
            """List calendar events."""
            return "events"

        @tool
        def send_email() -> str:
            """Send email."""
            return "sent"

        @tool
        def do_calendar_thing() -> str:
            """Calendar action."""
            return "unknown"

        self.assertTrue(is_read_only_tool(list_calendar_events))
        self.assertFalse(is_read_only_tool(send_email))
        self.assertFalse(is_read_only_tool(do_calendar_thing))
        self.assertFalse(
            is_read_only_tool(
                SimpleNamespace(
                    name="send_email",
                    metadata={"annotations": {"readOnlyHint": True}},
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
