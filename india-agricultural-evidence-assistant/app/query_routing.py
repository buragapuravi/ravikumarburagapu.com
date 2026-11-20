from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


_ATTENDANCE_TERMS = {"attended", "attendance", "participants", "participated", "women", "farmers"}
_APY_TERMS = {
    "area", "areas", "acreage", "production", "productions", "yield", "yields",
    "statistics", "statistic", "output",
}
_GENERIC_EVENT_TERMS = {"icar", "ciari", "field", "day", "event", "how", "many", "were", "what"}


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def route_query(question: str, workflow: str = "auto") -> dict[str, str | bool]:
    selected = workflow.casefold().strip()
    if selected == "text":
        return {"path": "text", "reason": "Text retrieval selected.", "require_current_advisory": False}
    if selected == "agentic":
        return {
            "path": "agentic",
            "reason": "LLM-planned, bounded research with local evidence search selected.",
            "require_current_advisory": False,
        }
    if selected == "statistics":
        return {"path": "statistics", "reason": "Structured DES query selected.", "require_current_advisory": False}
    if selected == "compare":
        return {"path": "statistics", "reason": "District comparison selected.", "require_current_advisory": False}
    if selected == "current_advisory":
        return {"path": "text", "reason": "Current-advisory validity check selected.", "require_current_advisory": True}

    terms = set(_normalized(question).split())
    if _ATTENDANCE_TERMS.intersection(terms) and re.search(
        r"\b(?:how many|number of|count|total|attendance)\b",
        question,
        re.IGNORECASE,
    ):
        return {
            "path": "extension",
            "reason": "Attendance and participation values use the structured indicator dataset.",
            "require_current_advisory": False,
        }
    asks_numeric = bool(
        re.search(r"\b(?:how much|how many|what was|compare|statistics|data|reported)\b", question, re.I)
    )
    if _APY_TERMS.intersection(terms) and asks_numeric:
        return {"path": "statistics", "reason": "Crop statistics are answered from structured DES records.", "require_current_advisory": False}
    if re.search(r"\bcurrent\b.*\badvisory\b|\badvisory\b.*\bcurrent\b", question, re.I):
        return {"path": "text", "reason": "Current-advisory validity check selected from question wording.", "require_current_advisory": True}
    return {"path": "text", "reason": "Research question routed to source-text retrieval.", "require_current_advisory": False}


def select_des_district_rows(
    question: str,
    rows: Sequence[Mapping[str, str]],
    filters: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    normalized_question = f" {_normalized(question)} "
    requested: dict[str, str] = {}
    for field in ("state", "district", "crop", "season", "agricultural_year"):
        values = sorted({row[field] for row in rows if row.get(field)}, key=len, reverse=True)
        for value in values:
            normalized_value = _normalized(value)
            if field == "agricultural_year":
                normalized_value = normalized_value.replace(" ", "")
                question_years = {
                    _normalized(item).replace(" ", "")
                    for item in re.findall(r"\b20\d{2}\s*[-/]\s*(?:20)?\d{2}\b", question)
                }
                if normalized_value in question_years:
                    requested[field] = value
                    break
            elif f" {normalized_value} " in normalized_question:
                requested[field] = value
                break

    active_filters = {
        key: value
        for key, value in (filters or {}).items()
        if value and key in {"state", "district", "crop", "season", "agricultural_year"}
    }
    active_filters.update(requested)
    return [
        dict(row)
        for row in rows
        if all(
            _normalized(row.get(key, "")) == _normalized(value)
            for key, value in active_filters.items()
        )
    ]


def match_event_names(question: str, event_names: Sequence[str]) -> list[str]:
    query_terms = set(_normalized(question).split()) - _GENERIC_EVENT_TERMS - _ATTENDANCE_TERMS
    if not query_terms:
        return []
    scores = {
        name: len(query_terms.intersection(set(_normalized(name).split())))
        for name in event_names
    }
    best = max(scores.values(), default=0)
    if best < 2:
        return []
    return [name for name, score in scores.items() if score == best]
