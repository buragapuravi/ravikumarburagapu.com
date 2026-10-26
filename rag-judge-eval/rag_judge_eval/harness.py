"""Input validation, annotation templates, and OpenAI-compatible judge calls."""

from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rag_judge_eval.metrics import LABELS, summarize_prompt
from rag_judge_eval.prompts import PROMPT_NAMES, build_prompt


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            required = ("id", "question_type", "question", "reference_answer")
            if any(not isinstance(case.get(key), str) or not case[key].strip() for key in required):
                raise ValueError(f"required fields must be non-empty strings: {required}")
            candidate = case.get("candidate_answer")
            passages = case.get("cited_passages")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("candidate_answer must be a non-empty string")
            if (
                not isinstance(passages, list)
                or not passages
                or any(not isinstance(text, str) or not text.strip() for text in passages)
            ):
                raise ValueError("cited_passages must be a non-empty list of strings")
            if case["id"] in seen:
                raise ValueError(f"duplicate id: {case['id']}")
            seen.add(case["id"])
            cases.append({**case, "cited_passages": passages})
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid case at {path}:{line_number}: {error}") from error
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    return cases


def write_annotation_template(cases: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "id",
        "question_type",
        "question",
        "reference_answer",
        "candidate_answer",
        "cited_passages",
        "human_answer_label",
        "human_citation_supported",
        "human_notes",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "id": case["id"],
                    "question_type": case["question_type"],
                    "question": case["question"],
                    "reference_answer": case["reference_answer"],
                    "candidate_answer": case["candidate_answer"],
                    "cited_passages": json.dumps(case["cited_passages"], ensure_ascii=False),
                    "human_answer_label": "",
                    "human_citation_supported": "",
                    "human_notes": "",
                }
            )


def load_annotations(path: Path, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {case["id"]: case for case in cases}
    annotations = {}
    with path.open(encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.DictReader(source), 2):
            case_id = row.get("id", "")
            if case_id not in by_id:
                raise ValueError(f"unknown annotation id at {path}:{line_number}: {case_id}")
            if case_id in annotations:
                raise ValueError(f"duplicate annotation id at {path}:{line_number}: {case_id}")
            for field in (
                "question_type",
                "question",
                "reference_answer",
                "candidate_answer",
            ):
                if row.get(field) != by_id[case_id][field]:
                    raise ValueError(
                        f"{field} changed for {case_id}; annotate the generated template "
                        "without changing case content"
                    )
            try:
                cited_passages = json.loads(row.get("cited_passages", ""))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid cited_passages JSON for {case_id}") from error
            if cited_passages != by_id[case_id]["cited_passages"]:
                raise ValueError(
                    f"cited_passages changed for {case_id}; annotate the generated template "
                    "without changing case content"
                )
            answer_label = (row.get("human_answer_label") or "").strip()
            citation_value = (row.get("human_citation_supported") or "").strip().lower()
            if answer_label and answer_label not in LABELS:
                raise ValueError(
                    f"invalid human_answer_label for {case_id}: choose {', '.join(LABELS)}"
                )
            if citation_value and citation_value not in ("true", "false"):
                raise ValueError(
                    f"invalid human_citation_supported for {case_id}: enter true or false"
                )
            annotations[case_id] = {
                "human_answer_label": answer_label or None,
                "human_citation_supported": (
                    citation_value == "true" if citation_value else None
                ),
                "human_notes": (row.get("human_notes") or "").strip(),
            }
    return annotations


def _call_judge(
    prompt: str, *, model: str, max_tokens: int, timeout: float
) -> tuple[str, dict[str, int]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run LLM judges")
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"judge request failed: {error}") from error
    try:
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return text, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("judge returned an invalid OpenAI-compatible response") from error


def _parse_judgment(text: str) -> dict[str, Any]:
    try:
        judgment = json.loads(text)
        if judgment.get("answer_label") not in LABELS:
            raise ValueError("answer_label is not a supported label")
        if not isinstance(judgment.get("citation_supported"), bool):
            raise ValueError("citation_supported must be boolean")
        if not isinstance(judgment.get("rationale"), str):
            raise ValueError("rationale must be a string")
        return judgment
    except (json.JSONDecodeError, AttributeError, ValueError) as error:
        raise RuntimeError(f"judge response did not match the required schema: {error}") from error


def run_evaluation(
    cases: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    model: str,
    input_cost_per_million: float,
    output_cost_per_million: float,
    human_hourly_rate: float | None,
    human_minutes_per_case: float | None,
    timeout: float = 90,
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run LLM judges")
    rows = []
    prompt_usage = {name: {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0} for name in PROMPT_NAMES}
    for case in cases:
        row = {**case, **annotations.get(case["id"], {})}
        for name in PROMPT_NAMES:
            text, usage = _call_judge(
                build_prompt(name, case), model=model, max_tokens=180, timeout=timeout
            )
            judgment = _parse_judgment(text)
            cost = (
                usage["prompt_tokens"] * input_cost_per_million
                + usage["completion_tokens"] * output_cost_per_million
            ) / 1_000_000
            prompt_usage[name]["prompt_tokens"] += usage["prompt_tokens"]
            prompt_usage[name]["completion_tokens"] += usage["completion_tokens"]
            prompt_usage[name]["cost"] += cost
            row[name] = {**judgment, "usage": usage, "cost": cost}
        rows.append(row)

    human_labeled = sum(
        row.get("human_answer_label") in LABELS
        and row.get("human_citation_supported") in (True, False)
        for row in rows
    )
    report: dict[str, Any] = {
        "model": model,
        "prompt_version": 1,
        "prompt_names": list(PROMPT_NAMES),
        "total_cases": len(rows),
        "human_labeled_cases": human_labeled,
        "prompts": {
            name: {
                **summarize_prompt(rows, name),
                "usage": {
                    key: value
                    for key, value in prompt_usage[name].items()
                    if key != "cost"
                },
                "cost_usd": prompt_usage[name]["cost"],
                "cost_per_1000_cases_usd": prompt_usage[name]["cost"]
                / len(rows)
                * 1000,
                "cost_per_1000_label_decisions_usd": prompt_usage[name]["cost"]
                / (len(rows) * 2)
                * 1000,
            }
            for name in PROMPT_NAMES
        },
        "human_label_cost": None,
        "cases": rows,
    }
    if human_hourly_rate is not None and human_minutes_per_case is not None:
        human_cost = human_labeled * human_minutes_per_case / 60 * human_hourly_rate
        report["human_label_cost"] = {
            "hourly_rate_usd": human_hourly_rate,
            "minutes_per_case": human_minutes_per_case,
            "labeled_cases": human_labeled,
            "cost_usd": human_cost,
            "cost_per_1000_labels_usd": (
                human_cost / human_labeled * 1000 if human_labeled else None
            ),
        }
    return report
