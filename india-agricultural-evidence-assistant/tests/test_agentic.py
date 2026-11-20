from __future__ import annotations

import json

import pytest

from app.agentic import _planned_queries, ask_agentic
from app.config import Settings
from app.models import Chunk, RetrievedChunk, SourceMetadata


def _evidence() -> RetrievedChunk:
    metadata = SourceMetadata(
        source_id="demo-source",
        filename="note.md",
        title="Research note",
        organization="Test institute",
        status="official international reference summary",
    )
    return RetrievedChunk(
        Chunk(
            chunk_id="chunk-1",
            text="The research note records a standing water observation for rice.",
            metadata=metadata,
            page_number=2,
            chunk_index=0,
        ),
        dense_score=0.8,
        lexical_score=0.9,
        fusion_score=0.05,
    )


def test_planner_queries_are_bounded_deduplicated_and_keep_original_question():
    queries = _planned_queries(
        "What does the research note say?",
        json.dumps({"queries": ["What observations are stated?", " what observations are stated? "]}),
    )

    assert queries == [
        "What does the research note say?",
        "What observations are stated?",
    ]


@pytest.mark.parametrize(
    "plan",
    [
        '{"queries":"not a list"}',
        '{"queries":["one","two","three"]}',
        '{"queries":[""]}',
        json.dumps({"queries": ["x" * 241]}),
    ],
)
def test_planner_rejects_malformed_or_unbounded_search_plans(plan: str):
    with pytest.raises(RuntimeError):
        _planned_queries("Research question", plan)


def test_agentic_rag_plans_searches_and_returns_citation_validated_answer(monkeypatch):
    from app import agentic

    calls = iter(
        [
            json.dumps({"queries": ["What observation is reported for rice?"]}),
            json.dumps(
                {
                    "answer": "The note reports a standing water observation for rice.",
                    "citations": ["demo-source|p. 2"],
                }
            ),
        ]
    )
    prompts = []

    def fake_request(config, prompt, system_prompt=""):
        prompts.append((prompt, system_prompt))
        return next(calls)

    retrieve_queries = []

    def fake_retrieve(**kwargs):
        retrieve_queries.append(kwargs["query"])
        return [_evidence()], {"reranker_state": "disabled"}

    monkeypatch.setattr(agentic, "_request", fake_request)
    monkeypatch.setattr("app.answer._request", fake_request)
    monkeypatch.setattr(agentic, "create_store", lambda config: object())
    monkeypatch.setattr(agentic, "retrieve", fake_retrieve)

    result = ask_agentic(
        "What does the research note say?",
        Settings(llm_provider="openai-compatible", llm_model="test-model"),
    )

    assert not result.abstained
    assert result.source_ids == ["demo-source"]
    assert retrieve_queries == [
        "What does the research note say?",
        "What observation is reported for rice?",
    ]
    assert result.diagnostics["agentic"]
    assert result.diagnostics["planner_queries"] == 2
    assert result.diagnostics["llm_calls"] == 2
    assert len(result.diagnostics["agent_steps"]) == 2
    assert "bounded retrieval steps" in result.provider_state
    assert prompts[0][1].startswith("You are a research planner")


def test_agentic_rag_requires_an_explicitly_configured_llm():
    with pytest.raises(ValueError, match="requires a configured LLM"):
        ask_agentic("What does the note say?", Settings(llm_provider="none"))


def test_agentic_rag_does_not_synthesize_when_evidence_gate_fails(monkeypatch):
    from app import agentic

    calls = []

    def fake_request(config, prompt, system_prompt=""):
        calls.append(system_prompt)
        return json.dumps({"queries": []})

    def fake_retrieve(**kwargs):
        return [], {"reranker_state": "disabled"}

    monkeypatch.setattr(agentic, "_request", fake_request)
    monkeypatch.setattr(agentic, "create_store", lambda config: object())
    monkeypatch.setattr(agentic, "retrieve", fake_retrieve)

    result = ask_agentic(
        "What does this research note say?",
        Settings(llm_provider="openai-compatible", llm_model="test-model"),
    )

    assert result.abstained
    assert result.diagnostics["llm_calls"] == 1
    assert not result.diagnostics["llm_called"]
    assert len(calls) == 1
