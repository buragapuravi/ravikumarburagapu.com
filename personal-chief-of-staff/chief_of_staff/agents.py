"""LangGraph supervisor and specialists with read-only MCP tool access."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chief_of_staff.workflows import SYSTEM_PROMPTS, Workflow

_READ_PREFIXES = ("get", "list", "search", "find", "read", "fetch", "query", "lookup", "retrieve")
_WRITE_VERBS = ("create", "send", "delete", "update", "edit", "write", "move", "cancel", "archive", "post")


def is_read_only_tool(tool: BaseTool) -> bool:
    """Fail closed: accept explicitly read-only annotations or familiar read verbs."""
    metadata = tool.metadata or {}
    annotations = metadata.get("annotations", metadata)
    normalized = tool.name.casefold().replace("-", "_")
    parts = normalized.split("_")
    if any(part in _WRITE_VERBS for part in parts):
        return False
    if isinstance(annotations, dict) and annotations.get("readOnlyHint") is True:
        return True
    return any(part.startswith(_READ_PREFIXES) for part in parts)


class _SupervisorDecision(BaseModel):
    workflow: Literal["scheduler", "email_triager", "meeting_prep", "daily_briefing"] = Field(
        description="The single specialist best suited to the user's request."
    )


class AssistantState(TypedDict):
    task: str
    workflow: str
    messages: Annotated[list, add_messages]


def build_graph(model: ChatOpenAI, tools: list[BaseTool]):
    """Build a supervisor-routed specialist graph with read-only MCP tools only."""
    supervisor = model.with_structured_output(_SupervisorDecision)
    graph = StateGraph(AssistantState)
    safe_tools = [tool for tool in tools if is_read_only_tool(tool)]
    tools_by_agent: dict[str, list[BaseTool]] = {}
    for workflow in SYSTEM_PROMPTS:
        terms = {
            "scheduler": ("calendar", "event", "schedule", "availability"),
            "email_triager": ("gmail", "email", "mail", "message", "inbox"),
            "meeting_prep": ("calendar", "event", "gmail", "email", "mail", "search", "notion", "task"),
            "daily_briefing": ("calendar", "event", "gmail", "email", "mail", "search", "notion", "task"),
        }[workflow]
        tools_by_agent[workflow] = [
            tool for tool in safe_tools if any(term in tool.name.casefold() for term in terms)
        ]

    def route_supervisor(state: AssistantState) -> dict:
        decision = supervisor.invoke(
            [
                SystemMessage(
                    content=(
                        "Route the request to exactly one specialist. Choose scheduler for finding "
                        "meeting times, email_triager for inbox/email, meeting_prep for an upcoming "
                        "meeting, otherwise daily_briefing."
                    )
                ),
                HumanMessage(content=state["task"]),
            ]
        )
        return {"workflow": decision.workflow}

    graph.add_node("supervisor", route_supervisor)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["workflow"],
        {workflow: workflow for workflow in SYSTEM_PROMPTS},
    )

    for workflow, prompt in SYSTEM_PROMPTS.items():
        specialist = model.bind_tools(tools_by_agent[workflow]) if tools_by_agent[workflow] else model

        def agent_node(state: AssistantState, llm=specialist, system_prompt=prompt) -> dict:
            answer = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
            return {"messages": [answer]}

        graph.add_node(workflow, agent_node)
        if tools_by_agent[workflow]:
            tool_node_name = f"{workflow}_read_tools"
            graph.add_node(tool_node_name, ToolNode(tools_by_agent[workflow]))
            graph.add_conditional_edges(
                workflow,
                lambda state: "tools" if state["messages"][-1].tool_calls else "done",
                {"tools": tool_node_name, "done": END},
            )
            graph.add_edge(tool_node_name, workflow)
        else:
            graph.add_edge(workflow, END)
    return graph.compile()


async def run_assistant(task: str, model: ChatOpenAI, tools: list[BaseTool]) -> dict[str, str]:
    if not task.strip():
        raise ValueError("task must not be empty")
    graph = build_graph(model, tools)
    result = await graph.ainvoke({"task": task, "workflow": "", "messages": [HumanMessage(content=task)]})
    response = result["messages"][-1].content
    if not isinstance(response, str):
        response = str(response)
    return {"workflow": result["workflow"], "response": response, "mode": "live"}
