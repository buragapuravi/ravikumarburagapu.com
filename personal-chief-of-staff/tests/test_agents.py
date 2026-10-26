import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from chief_of_staff.agents import build_graph


class StubRouter:
    def __init__(self, workflow: str) -> None:
        self.workflow = workflow

    def invoke(self, _messages):
        return SimpleNamespace(workflow=self.workflow)


class StubModel:
    def __init__(self, workflow: str = "daily_briefing") -> None:
        self.workflow = workflow
        self.bound_tool_names: list[list[str]] = []

    def with_structured_output(self, _schema):
        return StubRouter(self.workflow)

    def bind_tools(self, tools):
        self.bound_tool_names.append([tool.name for tool in tools])
        return self

    def invoke(self, _messages):
        return AIMessage(content="Prepared by the selected specialist.")


class AgentGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_supervisor_routes_request_to_specialist(self) -> None:
        result = await build_graph(StubModel("meeting_prep"), []).ainvoke(
            {
                "task": "Prepare me for my next meeting.",
                "workflow": "",
                "messages": [],
            }
        )
        self.assertEqual(result["workflow"], "meeting_prep")
        self.assertEqual(result["messages"][-1].content, "Prepared by the selected specialist.")

    async def test_only_read_tools_are_bound_to_agents(self) -> None:
        @tool
        def search_email() -> str:
            """Search email messages."""
            return "messages"

        @tool
        def send_email() -> str:
            """Send an email."""
            return "sent"

        model = StubModel("email_triager")
        await build_graph(model, [search_email, send_email]).ainvoke(
            {"task": "Triage my inbox", "workflow": "", "messages": []}
        )
        self.assertIn(["search_email"], model.bound_tool_names)
        self.assertTrue(all("send_email" not in names for names in model.bound_tool_names))


if __name__ == "__main__":
    unittest.main()
