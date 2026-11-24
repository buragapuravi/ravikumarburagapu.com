from __future__ import annotations

import json
import re
from dataclasses import replace
from urllib.parse import urlparse

import streamlit as st

from app.agentic import ask_agentic
from app.answer import split_answer_sentences
from app.config import settings
from app.engine import ask
from app.evaluate import evaluate_dataset
from app.ingest import create_store, ingest_paths
from app.preferences import load_ui_settings, save_ui_settings
from app.query_routing import match_event_names, route_query, select_des_district_rows
from app.structured_evidence import (
    event_attendance_summary,
    filter_records,
    load_des_district_records,
    load_extension_indicators,
    source_registry,
)

BASE_CONFIG = settings()
saved_settings_error = ""
try:
    SAVED_UI_SETTINGS = load_ui_settings(
        BASE_CONFIG.ui_settings_path,
        {
            "provider": BASE_CONFIG.llm_provider,
            "model": BASE_CONFIG.llm_model,
            "endpoint": BASE_CONFIG.llm_base_url,
        },
    )
except (OSError, RuntimeError, ValueError) as error:
    SAVED_UI_SETTINGS = {
        "provider": BASE_CONFIG.llm_provider,
        "model": BASE_CONFIG.llm_model,
        "endpoint": BASE_CONFIG.llm_base_url,
    }
    saved_settings_error = str(error)
st.set_page_config(
    page_title="Field Evidence Workbench",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stMetric"] {
        background: #f3f7f2; border: 1px solid #dfe9dc;
        padding: 14px 18px; border-radius: 12px;
    }
    [data-testid="stMetricLabel"] {color: #506453;}
    .eyebrow {
        color: #527a56; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;
    }
    .subtle {color: #637067; font-size: .94rem;}
    .branch-badge {
        display: inline-block; font-size: .7rem; font-weight: 700;
        letter-spacing: .09em; text-transform: uppercase;
        padding: 2px 8px; border-radius: 4px; margin-bottom: 4px;
    }
    .badge-text    {color: #2d7d46; border: 1px solid #2d7d46; background: #f0f9f2;}
    .badge-struct  {color: #1565c0; border: 1px solid #1565c0; background: #e8f0fe;}
    .badge-agent   {color: #6a1b9a; border: 1px solid #6a1b9a; background: #f3e5f5;}
    .badge-abstain {color: #c62828; border: 1px solid #c62828; background: #fce8e8;}
    .pipeline-flow {
        font-size: .82rem; color: #3d6b45;
        font-family: monospace; letter-spacing: .02em;
        background: #f3f7f2; border: 1px solid #dfe9dc;
        padding: 6px 12px; border-radius: 6px; margin-bottom: 8px;
    }
    .config-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #7a9280; margin: 0;}
    .config-value {font-size: .88rem; font-weight: 600; color: #2b4d32; margin: 0 0 8px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)
if saved_settings_error:
    st.error(f"Saved answer settings could not be loaded: {saved_settings_error}")

manifest: dict = {}
if BASE_CONFIG.manifest_path.exists():
    try:
        manifest = json.loads(BASE_CONFIG.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read the ingestion manifest: {error}")


def _scenario(
    question: str,
    crop: str = "Any crop",
    region: str = "Any region",
    season: str = "Any season",
    workflow: str = "",
) -> None:
    st.session_state["question_text"] = question
    st.session_state["crop_filter"] = crop
    st.session_state["state_filter"] = "Any state" if region == "Any region" else region
    st.session_state["district_filter"] = "Any district"
    st.session_state["season_filter"] = season
    if workflow:
        st.session_state["workflow_choice"] = workflow


@st.cache_data(show_spinner="Reading available evidence filters…")
def _corpus_facets() -> dict[str, object]:
    store = create_store(BASE_CONFIG)
    values = {"crop": set(), "state": set(), "district": set(), "season": set(), "status": set()}
    districts_by_state: dict[str, set[str]] = {}
    for chunk in store.all_chunks():
        metadata = chunk.metadata
        for facet in values:
            value = getattr(metadata, facet, "")
            if value:
                values[facet].add(value.strip())
        if metadata.state and metadata.district:
            districts_by_state.setdefault(metadata.state, set()).add(metadata.district)
    return {
        facet: sorted(
            {item.title() if facet in {"crop", "season"} else item for item in items},
            key=str.casefold,
        )
        for facet, items in values.items()
    } | {
        "district_by_state": {
            state: sorted(districts, key=str.casefold)
            for state, districts in districts_by_state.items()
        }
    }


@st.cache_data(show_spinner="Reading the indexed source catalog…")
def _source_catalog() -> list[dict[str, str]]:
    store = create_store(BASE_CONFIG)
    sources: dict[str, dict[str, str]] = {}
    for chunk in store.all_chunks():
        metadata = chunk.metadata
        row = sources.setdefault(
            metadata.source_id,
            {
                "Source": metadata.title,
                "Organization": metadata.organization,
                "Type": metadata.document_type,
                "Status": metadata.status,
                "Issued": metadata.issued_date or "Not stated",
                "Fetched": metadata.source_fetch_date or "Unknown",
                "Validity": (
                    f"{metadata.validity_start or 'unspecified'} to "
                    f"{metadata.validity_end or 'unspecified'}"
                ),
                "Region": ", ".join(filter(None, (metadata.district, metadata.state))) or "Not specified",
                "Crop": metadata.crop or "Not specified",
                "Source ID": metadata.source_id,
                "URL": metadata.source_url,
                "SHA-256": metadata.file_sha256,
                "Extraction": metadata.extraction_method,
                "Access/reuse note": metadata.license_access_note,
                "Chunks": "0",
            },
        )
        row["Chunks"] = str(int(row["Chunks"]) + 1)
    return sorted(sources.values(), key=lambda row: (row["Organization"], row["Source"]))


def _valid_option(key: str, options: list[str], fallback: str) -> list[str]:
    current = st.session_state.get(key)
    matching = next(
        (option for option in options if isinstance(current, str) and option.casefold() == current.casefold()),
        None,
    )
    if matching is not None:
        st.session_state[key] = matching
    else:
        st.session_state[key] = fallback
    return options


def _claim_citations(answer: str, evidence) -> list[tuple[str, list[str]]]:
    claims = []
    for claim in split_answer_sentences(answer):
        references = re.findall(r"\[([^\[\]]+?)\]", claim)
        linked_sources = []
        for reference in references:
            rendered_match = re.fullmatch(r"(.+), (p\. \d+|page unavailable)", reference)
            token_match = re.fullmatch(r"([a-zA-Z0-9._-]+)\|(p\.\s*\d+|n\.p\.)", reference)
            if rendered_match:
                title, page_label = rendered_match.groups()
                page_number = int(page_label.removeprefix("p. ")) if page_label.startswith("p.") else None
                source_id = ""
            elif token_match:
                source_id, page_label = token_match.groups()
                page_number = int(re.search(r"\d+", page_label).group()) if page_label != "n.p." else None
                title = ""
            else:
                continue
            for item in evidence:
                metadata = item.chunk.metadata
                if (
                    (metadata.title == title or metadata.source_id == source_id)
                    and item.chunk.page_number == page_number
                ):
                    linked_sources.append(
                        f"{metadata.title} · {metadata.organization} · "
                        f"{'p. ' + str(page_number) if page_number else 'page unavailable'}"
                    )
        claims.append((claim, list(dict.fromkeys(linked_sources))))
    return claims


def _district_yield_chart(rows: list[dict[str, str]]) -> None:
    st.vega_lite_chart(
        {
            "data": {
                "values": [
                    {
                        "district": row["district"],
                        "yield": float(row["yield_tonne_per_hectare"]),
                    }
                    for row in rows
                ]
            },
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {
                    "field": "district",
                    "type": "nominal",
                    "sort": "-y",
                    "axis": {"labelAngle": -45},
                },
                "y": {
                    "field": "yield",
                    "type": "quantitative",
                    "title": "Yield (tonnes/hectare)",
                },
                "tooltip": [
                    {"field": "district", "type": "nominal", "title": "District"},
                    {"field": "yield", "type": "quantitative", "title": "Yield (t/ha)"},
                ],
            },
        },
        use_container_width=True,
    )


def _latest_evaluation_report() -> tuple[dict, str] | None:
    project_dir = BASE_CONFIG.demo_dir.parent.parent
    reports = sorted(
        (project_dir / "reports").glob("evaluation-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return None
    report_path = reports[0]
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read evaluation report {report_path}: {error}") from error
    if not isinstance(payload, dict) or "retrieval_benchmark" not in payload:
        raise ValueError(f"Evaluation report has an unexpected schema: {report_path}")
    return payload, report_path.name


st.markdown('<p class="eyebrow">AGENTIC RAG · HYBRID RETRIEVAL · EVIDENCE-FIRST · INDIA AGRICULTURE</p>', unsafe_allow_html=True)
hero_left, hero_right = st.columns([3, 2])
with hero_left:
    st.title("Field Evidence Workbench")
    st.markdown(
        '<p class="subtle">Dense + BM25 → RRF → evidence gate → cited synthesis. '
        'Every claim traces back to a source passage; the system abstains when evidence is insufficient.</p>',
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown("")
    cfg1, cfg2 = st.columns(2)
    _provider_display = (
        "Extractive" if SAVED_UI_SETTINGS["provider"] == "none"
        else SAVED_UI_SETTINGS["provider"].replace("-", " ").title()
    )
    with cfg1:
        st.markdown(
            f'<p class="config-label">Framework</p>'
            f'<p class="config-value">{BASE_CONFIG.framework.title()} + {BASE_CONFIG.vector_backend.title()}</p>'
            f'<p class="config-label">Generation</p>'
            f'<p class="config-value">{_provider_display}</p>',
            unsafe_allow_html=True,
        )
    with cfg2:
        st.markdown(
            f'<p class="config-label">Retrieval</p>'
            f'<p class="config-value">Dense {BASE_CONFIG.dense_top_k} · BM25 {BASE_CONFIG.lexical_top_k}</p>'
            f'<p class="config-label">Embedding</p>'
            f'<p class="config-value">{BASE_CONFIG.embedding_model.split("/")[-1]}</p>',
            unsafe_allow_html=True,
        )

if not manifest.get("chunk_count"):
    st.warning("The evidence index is empty. Build the starter index before beginning a session.")
    if st.button("Build curated ICAR + FAO index", type="primary"):
        try:
            store = create_store(BASE_CONFIG)
            report = ingest_paths(
                [BASE_CONFIG.demo_dir, BASE_CONFIG.raw_dir],
                store,
                BASE_CONFIG.manifest_path,
                rebuild=True,
                framework=BASE_CONFIG.framework,
            )
        except (OSError, RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            _source_catalog.clear()
            st.success(
                f"Indexed {report['document_count']} documents and "
                f"{report['chunk_count']} evidence chunks. Refresh to open the workbench."
            )
    st.stop()

doc_count = manifest.get("document_count", 0)
chunk_count = manifest.get("chunk_count", 0)
official_count = sum(
    1 for row in manifest.get("documents", [])
    if "official" in str(row.get("status", "")).casefold()
)
synthetic_count = sum(
    1 for row in manifest.get("documents", [])
    if "synthetic" in str(row.get("status", "")).casefold()
)

metric_cols = st.columns(4)
metric_cols[0].metric("Indexed documents", doc_count)
metric_cols[1].metric("Searchable passages", chunk_count)
metric_cols[2].metric("Official-source summaries", official_count)
metric_cols[3].metric("Synthetic test fixtures", synthetic_count)

facets = _corpus_facets()
with st.sidebar:
    st.markdown("## Evidence scope")
    st.caption("Filters are generated from indexed metadata; unset filters search broadly.")
    crop_options = _valid_option("crop_filter", ["Any crop", *facets["crop"]], "Any crop")
    state_options = _valid_option("state_filter", ["Any state", *facets["state"]], "Any state")
    crop_filter = st.selectbox("Crop", crop_options, key="crop_filter")
    state_filter = st.selectbox("State / territory", state_options, key="state_filter")
    district_by_state = facets["district_by_state"]
    available_districts = (
        sorted(
            {
                district
                for state, districts in district_by_state.items()
                if state_filter == "Any state" or state.casefold() == state_filter.casefold()
                for district in districts
            },
            key=str.casefold,
        )
        if isinstance(district_by_state, dict)
        else []
    )
    district_options = _valid_option(
        "district_filter",
        ["Any district", *available_districts],
        "Any district",
    )
    district_filter = st.selectbox("District", district_options, key="district_filter")
    season_options = _valid_option("season_filter", ["Any season", *facets["season"]], "Any season")
    season_filter = st.selectbox("Season", season_options, key="season_filter")
    status_options = _valid_option("status_filter", ["Any source status", *facets["status"]], "Any source status")
    status_filter = st.selectbox("Source status", status_options, key="status_filter")
    filters = {
        "crop": "" if crop_filter == "Any crop" else crop_filter,
        "state": "" if state_filter == "Any state" else state_filter,
        "district": "" if district_filter == "Any district" else district_filter,
        "season": "" if season_filter == "Any season" else season_filter,
        "status": "" if status_filter == "Any source status" else status_filter,
    }

    with st.expander("Advanced answer settings"):
        provider_labels = {
            "none": "Extractive evidence mode",
            "ollama": "Local Ollama model",
            "openai-compatible": "OpenAI-compatible endpoint",
        }
        provider_values = list(provider_labels)
        provider_index = provider_values.index(SAVED_UI_SETTINGS["provider"])
        selected_provider = st.selectbox(
            "Generation provider",
            provider_values,
            index=provider_index,
            format_func=lambda value: provider_labels[value],
            help="Provider selection can be saved locally. API keys are never saved here.",
            key="provider_choice",
        )
        model_name = SAVED_UI_SETTINGS["model"]
        base_url = SAVED_UI_SETTINGS["endpoint"]
        api_key = ""
        if selected_provider != "none":
            model_name = st.text_input(
                "Model name",
                value=SAVED_UI_SETTINGS["model"],
                key="answer_model_name",
            )
            default_url = (
                SAVED_UI_SETTINGS["endpoint"]
                if SAVED_UI_SETTINGS["provider"] == selected_provider
                else (
                    "http://localhost:11434"
                    if selected_provider == "ollama"
                    else "https://api.openai.com"
                )
            )
            base_url = st.text_input(
                "Endpoint base URL",
                value=default_url,
                help="For OpenAI-compatible servers, enter the server root without /v1.",
                key="answer_endpoint",
            )
            if selected_provider == "openai-compatible":
                api_key = st.text_input(
                    "API key (optional)",
                    type="password",
                    help="Not saved by this app. Store persistent credentials in the ignored .env file.",
                )
        active_config = replace(
            BASE_CONFIG,
            llm_provider=selected_provider,
            llm_model=model_name.strip() or BASE_CONFIG.llm_model,
            llm_base_url=base_url.strip() or BASE_CONFIG.llm_base_url,
            llm_api_key=api_key or BASE_CONFIG.llm_api_key,
        )
        save_col, saved_col = st.columns([1, 2])
        with save_col:
            if st.button("Save settings", key="save_answer_settings"):
                try:
                    save_ui_settings(
                        BASE_CONFIG.ui_settings_path,
                        {
                            "provider": selected_provider,
                            "model": model_name.strip(),
                            "endpoint": base_url.strip(),
                        },
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.session_state["answer_settings_saved"] = True
        with saved_col:
            if st.session_state.pop("answer_settings_saved", False):
                st.success("Provider, model, and endpoint saved locally for future page loads.")
            else:
                st.caption("Use Save settings to keep these choices after refreshing the app.")
        if selected_provider == "none":
            st.info("No LLM call: responses show cited evidence excerpts.")
        elif selected_provider == "ollama":
            st.caption(
                "The Ollama service and selected model must already be running locally. "
                "Agentic mode uses one planning call and, when evidence passes, one answer call."
            )
        else:
            st.caption(
                "The question and retrieved passages are sent to the configured endpoint. "
                "Agentic mode uses one planning call and, when evidence passes, one answer call."
            )
        st.caption("Dense + BM25 → rank fusion → optional reranker → evidence and citation checks.")

with st.sidebar:
    with st.expander("Pipeline configuration"):
        _reranker_label = (
            f"`{BASE_CONFIG.reranker_model.split('/')[-1]}`"
            if BASE_CONFIG.reranker_model else "disabled"
        )
        st.markdown(
            f"**Retrieval** · Dense top-{BASE_CONFIG.dense_top_k} + BM25 top-{BASE_CONFIG.lexical_top_k}  \n"
            f"**Fusion** · RRF (lexical weight 2×)  \n"
            f"**Reranker** · {_reranker_label}  \n"
            f"**Evidence gate** · score ≥ `{BASE_CONFIG.evidence_min_score}` "
            f"· coverage ≥ `{BASE_CONFIG.evidence_min_coverage:.0%}`  \n"
            f"**Abstention** · conflicting evidence, missing context, expired advisory"
        )
        st.caption(
            f"Framework: {BASE_CONFIG.framework} + {BASE_CONFIG.vector_backend}  \n"
            f"Model: `{BASE_CONFIG.embedding_model.split('/')[-1]}`"
        )

st.markdown("### Demo scenarios")
st.caption("Six curated queries exercise different pipeline branches. Load one to pre-fill the question and filters, then inspect the full retrieval trace.")
scenario_cols = st.columns(2)
scenarios = [
    (
        "01 · Regional extension",
        "What topics were covered at ICAR-CIARI's lowland rice training in North and Middle Andaman?",
        "Rice",
        "Andaman and Nicobar Islands",
        "Any season",
        "ICAR-CIARI · training report",
    ),
    (
        "02 · Participation evidence",
        "How many farmers attended the Bloomsdale rice field day, and how many were women?",
        "Rice",
        "Andaman and Nicobar Islands",
        "Any season",
        "ICAR-CIARI · field day",
    ),
    (
        "03 · Agronomy concepts",
        "What are FAO's three principles of conservation agriculture?",
        "Any crop",
        "Any region",
        "Any season",
        "FAO · international technical context",
    ),
    (
        "04 · Conflict handling",
        "Which irrigation interval should cotton farmers in Maharashtra follow during kharif according to the demo notes?",
        "Cotton",
        "Maharashtra",
        "Kharif",
        "Synthetic fixture · expected abstention",
    ),
    (
        "05 · Rice soil evidence",
        "Which soil is good for rice crop?",
        "Rice",
        "Any region",
        "Any season",
        "IRRI · soil texture and water requirements",
    ),
    (
        "06 · Agentic research brief",
        "Compare rice soil water requirements, explain how groundwater depth changes the comparison, and identify what evidence is still needed for a local soil-suitability decision.",
        "Rice",
        "Any region",
        "Any season",
        "LLM plan · multi-query retrieval · cited synthesis",
        "Agentic research brief",
    ),
]
_SCENARIO_BRANCH: dict[str, tuple[str, str]] = {
    "01 · Regional extension":    ("TEXT · HYBRID",        "badge-text"),
    "02 · Participation evidence": ("STRUCTURED · ICAR",    "badge-struct"),
    "03 · Agronomy concepts":     ("TEXT · HYBRID",        "badge-text"),
    "04 · Conflict handling":     ("ABSTENTION · CONFLICT", "badge-abstain"),
    "05 · Rice soil evidence":    ("TEXT · QUERY EXPAND",   "badge-text"),
    "06 · Agentic research brief":("AGENTIC · MULTI-STEP",  "badge-agent"),
}
for offset in range(0, len(scenarios), 2):
    scenario_cols = st.columns(2)
    for column, (title, question, crop, region, season, subtitle, *workflow) in zip(
        scenario_cols,
        scenarios[offset : offset + 2],
    ):
        with column:
            branch_label, branch_cls = _SCENARIO_BRANCH.get(title, ("TEXT", "badge-text"))
            st.markdown(
                f'<span class="branch-badge {branch_cls}">{branch_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{title}**")
            st.caption(subtitle)
            st.button(
                "Load scenario",
                key=f"scenario_{title}",
                use_container_width=True,
                on_click=_scenario,
                args=(question, crop, region, season, workflow[0] if workflow else ""),
            )

workflow_labels = {
    "Automatic routing": "auto",
    "Find text evidence": "text",
    "Agentic research brief": "agentic",
    "Query DES statistics": "statistics",
    "Compare districts": "compare",
    "Check current advisory": "current_advisory",
}
workflow_label = st.selectbox(
    "Research workflow",
    list(workflow_labels),
    help="Agentic mode requires an LLM provider; it plans bounded searches over the local corpus before cited synthesis.",
    key="workflow_choice",
)
if workflow_labels[workflow_label] == "agentic":
    if active_config.llm_provider == "none":
        st.warning(
            "Agentic mode needs an LLM. Open **Advanced answer settings** in the sidebar "
            "and select a running Ollama model or configured OpenAI-compatible endpoint."
        )
    else:
        st.info(
            "The LLM plans up to two focused follow-up searches. The app always includes "
            "the original question, runs read-only hybrid search on the local index, then "
            "validates the synthesized answer's citations."
        )

st.markdown("### Research question")
st.caption("Questions route automatically to hybrid text retrieval, structured indicators, or agentic multi-step planning based on the selected workflow.")
with st.form("research_query", clear_on_submit=False):
    question = st.text_area(
        "Ask about the indexed evidence",
        key="question_text",
        height=100,
        placeholder="Example: What did the ICAR-CIARI training cover, and what did the source report about participation?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Run research workflow",
        type="primary",
        use_container_width=True,
    )

if submitted and not question.strip():
    st.warning("Enter a research question before running the workflow.")
elif submitted:
    try:
        route = route_query(question.strip(), workflow_labels[workflow_label])
        st.session_state.pop("structured_result", None)
        st.session_state.pop("query_result", None)
        st.session_state["query_route"] = route
        st.session_state["last_question"] = question.strip()
        st.session_state["last_filters"] = filters
        if route["path"] == "statistics":
            with st.spinner("Querying and validating source-linked DES records…"):
                district_records = load_des_district_records()
                matching_rows = select_des_district_rows(question.strip(), district_records, filters)
            st.session_state["structured_result"] = {
                "kind": "des",
                "rows": matching_rows,
                "comparison": route["reason"] == "District comparison selected."
                or bool(re.search(r"\bcompare|comparison|across districts\b", question, re.I)),
            }
        elif route["path"] == "extension":
            with st.spinner("Matching the question to verified structured indicators…"):
                indicator_rows = load_extension_indicators()
                event_names = sorted({row["event_name"] for row in indicator_rows})
                matches = match_event_names(question.strip(), event_names)
                st.session_state["structured_result"] = {
                    "kind": "extension",
                    "events": matches,
                    "rows": indicator_rows,
                }
        elif route["path"] == "agentic":
            with st.spinner("Planning research steps, searching local evidence, and validating a cited synthesis…"):
                result = ask_agentic(
                    question.strip(),
                    active_config,
                    filters,
                    require_current_advisory=bool(route["require_current_advisory"]),
                )
            st.session_state["query_result"] = result
        else:
            with st.spinner("Searching sources, fusing rankings, and checking evidence…"):
                result = ask(
                    question.strip(),
                    active_config,
                    filters,
                    require_current_advisory=bool(route["require_current_advisory"]),
                )
            st.session_state["query_result"] = result
    except (RuntimeError, ValueError, OSError) as error:
        st.error(f"Research query failed: {error}")

st.divider()
route = st.session_state.get("query_route")
structured_result = st.session_state.get("structured_result")
result = st.session_state.get("query_result")
if structured_result is not None:
    st.markdown(f"## Structured research result  \n*{st.session_state.get('last_question', '')}*")
    if route:
        st.caption(f"Route: {route['reason']}")
    if structured_result["kind"] == "des":
        rows = structured_result["rows"]
        if not rows:
            st.warning("No matching rows are in the loaded DES extract. Broaden the filters or choose a supported year, crop, and region.")
        else:
            rows = sorted(rows, key=lambda row: float(row["yield_tonne_per_hectare"]), reverse=True)
            top = rows[0]
            st.caption(
                f"DES · {top['crop']} · {top['season']} · {top['agricultural_year']} · "
                f"{len(rows)} matching district row{'s' if len(rows) != 1 else ''}"
            )
            if len(rows) == 1:
                metrics = st.columns(3)
                metrics[0].metric("Area", f"{float(top['area_hectares']):,.0f} ha")
                metrics[1].metric("Production", f"{float(top['production_tonnes']):,.0f} tonnes")
                metrics[2].metric("Yield", f"{float(top['yield_tonne_per_hectare']):.2f} t/ha")
            elif structured_result["comparison"]:
                _district_yield_chart(rows)
            st.dataframe(
                [
                    {
                        "District": row["district"],
                        "Area (ha)": float(row["area_hectares"]),
                        "Production (tonnes)": float(row["production_tonnes"]),
                        "Yield (tonnes/ha)": float(row["yield_tonne_per_hectare"]),
                        "Estimate status": row["estimate_status"],
                        "Source row SHA-256": row["source_record_sha256"],
                    }
                    for row in rows
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.warning(top["caveat"])
            st.link_button("Open official DES report portal", top["source_url"])
            with st.expander("Dataset provenance and query details"):
                st.caption(
                    f"Filters: {top['query_filters']} · Retrieved: {top['retrieved_at']} · "
                    f"Extraction: {top['extraction_method']}"
                )
                for row in rows:
                    st.caption(f"{row['district']}: `{row['source_row_canonical']}`")
    else:
        events = structured_result["events"]
        if len(events) == 1:
            summary = event_attendance_summary(structured_result["rows"], events[0])
            st.markdown(f"### {summary['event_name']}")
            metrics = st.columns(4)
            metrics[0].metric("Participants reported", summary["participants_total"])
            if "participants_women_reported" in summary:
                metrics[1].metric("Women reported", summary["participants_women_reported"])
                metrics[2].metric("Men reported", summary["participants_men_reported"])
                metrics[3].metric("Women share", f"{summary['women_share_percent']:.1f}%")
            st.caption(f"{summary['source_title']} · source ID `{summary['source_id']}`")
            st.link_button("Open original ICAR source", summary["source_url"])
            st.warning(summary["caveat"])
        elif events:
            st.info("Several events match. Add the event name or crop to disambiguate.")
            st.write(events)
        else:
            st.info("Name a specific event (for example, Bloomsdale rice field day) to query its structured indicators.")
            st.write(sorted({row["event_name"] for row in structured_result["rows"]}))
elif result is not None:
    _last_q = st.session_state.get('last_question', '')
    st.markdown("## Research brief")
    st.markdown(f"*{_last_q}*")
    if route:
        _path = route.get("path", "text").upper()
        _path_cls = {
            "TEXT": "badge-text", "AGENTIC": "badge-agent",
            "STATISTICS": "badge-struct", "EXTENSION": "badge-struct",
        }.get(_path, "badge-text")
        st.markdown(
            f'<span class="branch-badge {_path_cls}">{_path}</span> '
            f'<span class="subtle">{route["reason"]}</span>',
            unsafe_allow_html=True,
        )
    if result.diagnostics.get("agentic"):
        _planner_q = result.diagnostics.get('planner_queries', 0)
        _llm_calls = result.diagnostics.get('llm_calls', 0)
        _steps = result.diagnostics.get('agent_steps', [])
        st.info(
            f"**Agentic RAG** · {_planner_q} planner quer{'y' if _planner_q == 1 else 'ies'} "
            f"→ {len(_steps)} hybrid retrieval step{'s' if len(_steps) != 1 else ''} "
            f"→ {_llm_calls} LLM call{'s' if _llm_calls != 1 else ''} total  \n"
            f"Original question always searched · at most 2 follow-up queries · read-only local index"
        )
    gate = result.gate
    if result.abstained:
        _gate_reason = gate.reason if gate else "Insufficient evidence"
        st.error(f"**Evidence gate: Abstain** — {_gate_reason}")
        if gate and gate.missing_context:
            st.caption(f"Missing context: {' · '.join(gate.missing_context)}")
        st.warning(result.answer)
    else:
        _answer_mode = "model-assisted synthesis" if result.diagnostics.get("llm_called") else "extractive evidence mode"
        st.success(f"**Evidence gate: Pass** — {_answer_mode}")
        if gate:
            st.caption(f"Gate reason: {gate.reason}")
        st.markdown(result.answer)
        st.caption(f"Provider: {result.provider_state}")

    confidence_cols = st.columns(4)
    confidence_cols[0].metric("Gate decision", "Pass" if gate and gate.answerable else "Abstain")
    confidence_cols[1].metric("Term coverage", f"{gate.coverage:.0%}" if gate else "—")
    confidence_cols[2].metric("Top retrieval score", f"{gate.score:.3f}" if gate else "—")
    confidence_cols[3].metric("Evidence passages", len(result.evidence))
    st.caption(
        "Term coverage and retrieval score are diagnostic pipeline signals, not answer confidence. "
        "The gate also checks requested topic, source status, and advisory validity dates."
    )

    claims_tab, answer_tab, trace_tab = st.tabs(["Claims & citations", "Sources & evidence", "Retrieval trace"])
    with claims_tab:
        if result.abstained:
            st.info("No answer claims were produced because the retrieved evidence was insufficient.")
            if gate and gate.missing_context:
                st.markdown("**Evidence needed**")
                for missing in gate.missing_context:
                    st.markdown(f"- {missing}")
        else:
            claims = _claim_citations(result.answer, result.evidence)
            if not claims:
                st.info("No answer claims were returned.")
            for claim, linked_sources in claims:
                st.markdown(claim)
                if linked_sources:
                    for source_reference in linked_sources:
                        st.caption(f"Supported by: {source_reference}")
                else:
                    st.warning("No retrieved source/page citation could be mapped to this statement.")
    with answer_tab:
        if not result.evidence:
            st.info("No passages passed the active corpus filters. Broaden the scope or choose another question.")
        for rank, item in enumerate(result.evidence, start=1):
            source = item.chunk.metadata
            page = f"p. {item.chunk.page_number}" if item.chunk.page_number else "page unavailable"
            title = f"{rank}. {source.title}"
            with st.expander(title, expanded=(rank == 1)):
                source_col, score_col = st.columns([3, 1])
                with source_col:
                    st.caption(
                        f"{source.organization} · {source.document_type} · "
                        f"{source.issued_date or 'date not stated'} · {source.status}"
                    )
                    st.caption(
                        f"Scope: {source.crop or 'crop unspecified'} · "
                        f"{', '.join(filter(None, (source.district, source.state))) or 'region unspecified'} · {page}"
                    )
                    st.caption(
                        f"Validity: {source.validity_start or 'unspecified'} to "
                        f"{source.validity_end or 'unspecified'} · "
                        f"Source fetched: {source.source_fetch_date or 'unknown'}"
                    )
                with score_col:
                    st.markdown(
                        f"**Rank {rank}**  \n"
                        f"RRF `{item.fusion_score:.4f}`  \n"
                        f"Dense `{item.dense_score:.3f}` · BM25 `{item.lexical_score:.3f}`"
                    )
                st.markdown("> " + item.chunk.text.replace("\n", "\n> "))
                if source.source_url and urlparse(source.source_url).scheme in {"http", "https"}:
                    st.link_button("Open original source", source.source_url)
                st.caption(f"Stable source ID: `{source.source_id}` · Citation page: {page}")
                if source.file_sha256:
                    st.caption(
                        f"Local file SHA-256: `{source.file_sha256}` · "
                        f"Extraction: {source.extraction_method}"
                    )
                if source.license_access_note:
                    st.caption(f"Access / reuse: {source.license_access_note}")
                if source.extraction_warnings:
                    st.warning("; ".join(source.extraction_warnings))
    with trace_tab:
        _reranker_stage = (
            f"reranker top-{BASE_CONFIG.rerank_top_k}"
            if BASE_CONFIG.reranker_model else "reranker disabled"
        )
        st.markdown(
            f'<p class="pipeline-flow">'
            f'Dense (k={BASE_CONFIG.dense_top_k}) + BM25 (k={BASE_CONFIG.lexical_top_k}) '
            f'→ RRF (lexical 2×) → {_reranker_stage} → evidence gate → synthesis'
            f'</p>',
            unsafe_allow_html=True,
        )
        diagnostics = result.diagnostics
        if diagnostics.get("agentic"):
            st.markdown("**Visible agent actions** · read-only local hybrid search only")
            st.caption(
                "Tool calls, queries, and evidence counts are shown here. "
                "The model's internal reasoning is not exposed. "
                "The agent cannot browse the web, execute code, or modify data."
            )
            for step_number, step in enumerate(diagnostics.get("agent_steps", []), start=1):
                with st.expander(
                    f"Step {step_number} · {step['action']} · {step['evidence_count']} passages retrieved",
                    expanded=True,
                ):
                    st.markdown(f"**Query:** `{step['query']}`")
                    st.caption(f"Sources: {', '.join(step['source_ids']) or 'none'}")
                    st.caption(f"Reranker: {step['reranker_state']}")
        if diagnostics.get("query_expansion"):
            st.caption(f"Query expansion applied: {diagnostics['query_expansion']}")
        trace_cols = st.columns(2)
        with trace_cols[0]:
            st.markdown("**Stage 1a · Dense retrieval** (embedding similarity)")
            dense = diagnostics.get("dense", [])
            if dense:
                st.dataframe(dense, hide_index=True, use_container_width=True)
            else:
                st.caption("No dense candidates.")
        with trace_cols[1]:
            st.markdown("**Stage 1b · Lexical retrieval** (BM25)")
            lexical = diagnostics.get("lexical", [])
            if lexical:
                st.dataframe(lexical, hide_index=True, use_container_width=True)
            else:
                st.caption("No lexical candidates.")
        st.markdown("**Stage 2 · RRF fusion + evidence gate**")
        _gate_label = "PASS" if gate and gate.answerable else "ABSTAIN"
        _gate_cls = "badge-text" if gate and gate.answerable else "badge-abstain"
        _gate_reason = gate.reason if gate else "not run"
        st.markdown(
            f'<span class="branch-badge {_gate_cls}">{_gate_label}</span> {_gate_reason}',
            unsafe_allow_html=True,
        )
        fusion_data = diagnostics.get("fusion", {})
        if fusion_data:
            st.json(fusion_data)
        st.caption(
            f"Reranker: {diagnostics.get('reranker_state', 'not run')} · "
            f"Citation validation: "
            f"{'applied to generated answer' if diagnostics.get('llm_called') else 'no generated claims'} · "
            f"LLM called: {diagnostics.get('llm_called', False)}"
        )
elif not st.session_state.get("question_text", "").strip():
    st.info("Load a scenario or enter a research question to start an evidence-backed workflow.")
else:
    st.info("Review the scope, then select **Retrieve and analyze evidence** to run this query.")

with st.expander("Structured evidence explorer · deterministic queries"):
    st.markdown(
        "Numeric observations are queried separately from vector search. Values stay attached "
        "to their reporting period, units, geography, original source, and provenance hash."
    )
    event_tab, apy_tab = st.tabs(["ICAR extension indicators", "DES district comparison"])
    try:
        event_rows = load_extension_indicators()
        apy_rows = load_des_district_records()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Structured evidence could not be validated: {error}")
    else:
        with event_tab:
            event_names = sorted({row["event_name"] for row in event_rows})
            selected_event = st.selectbox("Reported event", event_names, key="indicator_event")
            event = event_attendance_summary(event_rows, selected_event)
            indicator_rows = filter_records(event_rows, event_name=selected_event)
            if "participants_women_reported" in event:
                total_col, women_col, men_col, share_col = st.columns(4)
                total_col.metric("Participants reported", event["participants_total"])
                women_col.metric("Women reported", event["participants_women_reported"])
                men_col.metric("Men reported", event["participants_men_reported"])
                share_col.metric("Women share", f"{event['women_share_percent']:.1f}%")
            else:
                st.metric("Villages represented", event["participants_total"])
            st.caption(
                f"{indicator_rows[0]['organization']} · {indicator_rows[0]['event_date']} · "
                f"{indicator_rows[0]['district']}, {indicator_rows[0]['state']} · "
                f"{indicator_rows[0]['temporal_scope']}"
            )
            st.warning(event["caveat"])
            st.dataframe(
                indicator_rows,
                hide_index=True,
                use_container_width=True,
                column_config={"source_url": st.column_config.LinkColumn("Original source")},
            )
            st.caption(
                "The women's share is calculated deterministically from the source-reported "
                "women and total counts; it is not an independently reported percentage."
            )

        with apy_tab:
            st.caption("District-wise values from one official DES Punjab Wheat/Rabi 2022–2023 report view.")
            states = sorted({row["state"] for row in apy_rows}, key=str.casefold)
            selected_state = st.selectbox("State", states, key="apy_state")
            state_rows = filter_records(apy_rows, state=selected_state)
            crops = sorted({row["crop"] for row in state_rows}, key=str.casefold)
            selected_crop = st.selectbox("Crop", crops, key="apy_crop")
            crop_rows = filter_records(state_rows, crop=selected_crop)
            seasons = sorted({row["season"] for row in crop_rows}, key=str.casefold)
            selected_season = st.selectbox("Season", seasons, key="apy_season")
            season_rows = filter_records(crop_rows, season=selected_season)
            years = sorted({row["agricultural_year"] for row in season_rows})
            selected_year = st.selectbox("Agricultural year", years, key="apy_year")
            year_rows = filter_records(season_rows, agricultural_year=selected_year)
            districts = sorted({row["district"] for row in year_rows}, key=str.casefold)
            selected_districts = st.multiselect(
                "Districts to compare",
                districts,
                default=districts,
                key="apy_districts",
            )
            comparison_rows = [
                row for row in year_rows if row["district"] in selected_districts
            ]
            if not comparison_rows:
                st.info("Select at least one district to compare.")
            else:
                ordered_rows = sorted(
                    comparison_rows,
                    key=lambda row: float(row["yield_tonne_per_hectare"]),
                    reverse=True,
                )
                top_cols = st.columns(3)
                top_cols[0].metric("Districts shown", len(ordered_rows))
                top_cols[1].metric("Highest reported yield", f"{ordered_rows[0]['yield_tonne_per_hectare']} t/ha")
                top_cols[2].metric("Lowest reported yield", f"{ordered_rows[-1]['yield_tonne_per_hectare']} t/ha")
                _district_yield_chart(ordered_rows)
                st.dataframe(
                    [
                        {
                            "District": row["district"],
                            "Area (ha)": float(row["area_hectares"]),
                            "Production (tonnes)": float(row["production_tonnes"]),
                            "Yield (tonnes/ha)": float(row["yield_tonne_per_hectare"]),
                            "Estimate status": row["estimate_status"],
                            "Row SHA-256": row["source_record_sha256"],
                        }
                        for row in ordered_rows
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                st.link_button("Open DES report portal", ordered_rows[0]["source_url"])
                st.caption(
                    f"Query: {ordered_rows[0]['query_filters']} · Retrieved {ordered_rows[0]['retrieved_at']} · "
                    f"Hash is over the visible normalized report row, not a downloaded DES file."
                )
                st.warning(ordered_rows[0]["caveat"])
            st.caption(
                "Historical official report values are not forecasts. The portal view did not state "
                "estimate status; verify against a downloaded official report before reuse."
            )

with st.expander("Corpus registry & data boundaries"):
    if st.button("Refresh source catalog", key="refresh_catalog"):
        _source_catalog.clear()
    try:
        catalog = _source_catalog()
    except (RuntimeError, ValueError, OSError) as error:
        st.error(f"Could not read the indexed source catalog: {error}")
    else:
        st.dataframe(
            catalog,
            hide_index=True,
            use_container_width=True,
            column_config={"URL": st.column_config.LinkColumn("Original source", display_text="Open")},
        )
    st.caption(
        "The corpus contains curator-written summaries of ICAR and FAO pages plus clearly "
        "labeled synthetic evaluation fixtures. Summaries are not full source reproductions. "
        "FAO references provide international context, not locally validated recommendations. "
        "Retrieved text is untrusted; this workbench is not an official advisory service."
    )
    try:
        registered_sources = source_registry()
    except (RuntimeError, ValueError) as error:
        st.error(f"Could not read the official-source registry: {error}")
    else:
        st.markdown("**Official-source acquisition registry**")
        st.dataframe(
            registered_sources,
            hide_index=True,
            use_container_width=True,
            column_config={"url": st.column_config.LinkColumn("Source URL")},
        )
        st.caption("Candidate sources are discovery records only; they are not indexed until content is verified and ingested.")
    failed_documents = len(manifest.get("failed_documents", []))
    st.caption(
        f"Index state: {manifest.get('document_count', 0)} documents · "
        f"{manifest.get('chunk_count', 0)} chunks · {failed_documents} ingestion failures."
    )

with st.expander("Evaluation & quality signals"):
    st.caption(
        "Run the bundled answerability, citation, and retrieval checks on demand. "
        "These small curated benchmarks are diagnostics, not a guarantee of field accuracy."
    )
    if st.button("Run evaluation suite", key="run_evaluation"):
        project_dir = BASE_CONFIG.demo_dir.parent.parent
        try:
            with st.spinner("Running answerability, citation, and retrieval checks…"):
                json_path, _, report = evaluate_dataset(
                    project_dir / "data/evaluation/questions.json",
                    project_dir / "reports",
                )
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Evaluation failed: {error}")
        else:
            st.session_state["latest_evaluation"] = (report, json_path.name)

    latest = st.session_state.get("latest_evaluation")
    if latest is None:
        try:
            latest = _latest_evaluation_report()
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not load the latest evaluation report: {error}")
    if latest is None:
        st.info("No evaluation report is available yet. Run the suite above or use `python -m app.evaluate`.")
    else:
        report, report_name = latest
        retrieval_metrics = report["retrieval_benchmark"]["aggregate"]
        deterministic = report["deterministic_checks"]
        confusion = report["confusion_matrix"]
        quality_cols = st.columns(4)
        quality_cols[0].metric("Retrieval recall@5", f"{retrieval_metrics['recall@5']:.2f}")
        quality_cols[1].metric("Retrieval nDCG@5", f"{retrieval_metrics['ndcg@5']:.2f}")
        quality_cols[2].metric("Citation coverage", f"{deterministic['mean_citation_coverage']:.0%}")
        quality_cols[3].metric(
            "Abstention precision",
            f"{confusion['abstention_precision']:.0%}",
        )
        st.caption(
            f"Latest report: {report_name} · generated {report.get('generated_at', 'timestamp unavailable')} · "
            f"{report['retrieval_benchmark'].get('query_count', 0)} retrieval questions"
        )
        st.caption(
            f"Source IDs valid: {deterministic['all_returned_source_ids_valid']} · "
            f"Unanswerable questions abstained: {deterministic['unanswerable_abstentions']} · "
            "Citation coverage uses answerable responses only. Small curated set; review failures and source changes."
        )
