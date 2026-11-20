from dataclasses import replace
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.models import (
    AnswerResult,
    Chunk,
    EvidenceDecision,
    RetrievedChunk,
    SourceMetadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loading_field_scenario_populates_question_and_scope() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)

    assert not app.exception
    assert app.text_area[0].value == ""
    submit = next(item for item in app.button if item.label == "Run research workflow")
    assert not submit.disabled

    question = "How many farmers attended the Bloomsdale rice field day, and how many were women?"
    scenario = next(item for item in app.button if item.key == "scenario_02 · Participation evidence")
    scenario.click().run()

    assert not app.exception
    assert app.text_area[0].value == question
    selected_crop = next(item for item in app.selectbox if item.key == "crop_filter")
    selected_region = next(item for item in app.selectbox if item.key == "state_filter")
    assert selected_crop.value == "Rice"
    assert selected_region.value == "Andaman and Nicobar Islands"
    assert not next(item for item in app.button if item.label == "Run research workflow").disabled


def test_loading_rice_soil_scenario_populates_the_research_question() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)

    scenario = next(item for item in app.button if item.key == "scenario_05 · Rice soil evidence")
    scenario.click().run()

    assert not app.exception
    assert app.text_area[0].value == "Which soil is good for rice crop?"
    selected_crop = next(item for item in app.selectbox if item.key == "crop_filter")
    assert selected_crop.value == "Rice"


def test_agentic_scenario_selects_the_agentic_workflow() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)

    scenario = next(item for item in app.button if item.key == "scenario_06 · Agentic research brief")
    scenario.click().run()

    assert not app.exception
    assert app.text_area[0].value.startswith("Compare rice soil water requirements")
    workflow = next(item for item in app.selectbox if item.key == "workflow_choice")
    assert workflow.value == "Agentic research brief"
    assert any("Agentic mode needs an LLM" in item.value for item in app.warning)


def test_save_answer_settings_persists_provider_model_and_endpoint(tmp_path, monkeypatch) -> None:
    from app.config import settings

    settings_path = tmp_path / "ui-settings.json"
    original_settings = settings
    monkeypatch.setattr(
        "app.config.settings",
        lambda: replace(original_settings(), ui_settings_path=settings_path),
    )

    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)
    provider = next(item for item in app.selectbox if item.key == "provider_choice")
    provider.select("ollama").run()
    next(item for item in app.button if item.key == "save_answer_settings").click().run()

    assert not app.exception
    assert settings_path.exists()
    assert any("saved locally" in item.value for item in app.success)
    reloaded_app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)
    assert not reloaded_app.exception
    assert next(item for item in reloaded_app.selectbox if item.key == "provider_choice").value == "ollama"
    assert next(item for item in reloaded_app.text_input if item.key == "answer_model_name").value == "llama3.1:8b"
    assert next(item for item in reloaded_app.text_input if item.key == "answer_endpoint").value == "http://localhost:11434"


def test_find_evidence_renders_answer_and_evidence(monkeypatch) -> None:
    source = SourceMetadata(
        source_id="icar-test-source",
        filename="icar-test.md",
        title="ICAR test source",
        organization="ICAR",
        source_url="https://icar.org.in/test",
        crop="rice",
        status="official",
    )
    evidence = RetrievedChunk(
        chunk=Chunk(
            chunk_id="test-chunk",
            text="The ICAR report records a farmer field day.",
            metadata=source,
            page_number=None,
            chunk_index=0,
        ),
        dense_score=0.9,
        lexical_score=0.8,
        fusion_score=0.05,
    )
    result = AnswerResult(
        answer="The report records a farmer field day. [icar-test-source|n.p.]",
        abstained=False,
        abstention_reason="",
        source_ids=["icar-test-source"],
        evidence=[evidence],
        gate=EvidenceDecision(True, "Test evidence passed.", 1.0, 0.9),
        provider_state="Test response",
    )
    monkeypatch.setattr("app.engine.ask", lambda *args, **kwargs: result)

    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)
    app.text_area[0].set_value("What does the report record?")
    next(item for item in app.button if item.label == "Run research workflow").click().run()

    assert not app.exception
    assert any("farmer field day" in item.value for item in app.markdown)
    assert any(item.label == "1. ICAR test source" for item in app.expander)
    assert any("Supported by" in item.value for item in app.caption)


def test_abstention_displays_evidence_gaps_without_claim_citation_warnings(monkeypatch) -> None:
    result = AnswerResult(
        answer="Insufficient evidence: No pH threshold is available.",
        abstained=True,
        abstention_reason="The source does not cover the requested soil property.",
        evidence=[],
        gate=EvidenceDecision(
            False,
            "The source does not cover the requested soil property.",
            0.8,
            1.0,
            ("source covering pH",),
        ),
    )
    monkeypatch.setattr("app.engine.ask", lambda *args, **kwargs: result)

    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)
    app.text_area[0].set_value("What is the optimal soil pH for rice?")
    next(item for item in app.button if item.label == "Run research workflow").click().run()

    assert not app.exception
    assert any("No answer claims were produced" in item.value for item in app.info)
    assert any("source covering pH" in item.value for item in app.markdown)
    assert not any("No retrieved source/page citation could be mapped" in item.value for item in app.warning)


def test_automatic_workflow_routes_district_statistics_to_comparison():
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)
    app.text_area[0].set_value("Compare wheat yield across Punjab districts")
    next(item for item in app.button if item.label == "Run research workflow").click().run()

    assert not app.exception
    assert any("Structured research result" in item.value for item in app.markdown)
    assert any("23 matching district rows" in item.value for item in app.caption)


def test_automatic_workflow_routes_event_counts_to_structured_indicators():
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)
    app.text_area[0].set_value(
        "How many farmers attended the Bloomsdale rice field day, and how many were women?"
    )
    next(item for item in app.button if item.label == "Run research workflow").click().run()

    assert not app.exception
    assert any("ICAR-CIARI Bloomsdale rice field day" in item.value for item in app.markdown)
    assert any(metric.label == "Women share" for metric in app.metric)
