import csv
import json

import pytest

from rag_judge_eval.harness import (
    load_annotations,
    load_cases,
    run_evaluation,
    write_annotation_template,
)
from rag_judge_eval.metrics import summarize_prompt
from rag_judge_eval.prompts import PROMPT_NAMES, build_prompt


def _case():
    return {
        "id": "demo-1",
        "question_type": "numeric",
        "question": "How many retries are allowed?",
        "reference_answer": "Three retries are allowed.",
        "candidate_answer": "Three retries are allowed.",
        "cited_passages": ["The client allows three retries."],
    }


def test_case_loader_rejects_duplicate_ids_and_invalid_passages(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(_case()) + "\n" + json.dumps(_case()) + "\n")
    with pytest.raises(ValueError, match="duplicate id"):
        load_cases(cases)

    broken = {**_case(), "cited_passages": []}
    cases.write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="cited_passages"):
        load_cases(cases)


def test_review_template_starts_unlabeled_and_round_trips(tmp_path):
    case = _case()
    review = tmp_path / "nested" / "review.csv"
    write_annotation_template([case], review)
    with review.open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["human_answer_label"] == ""
    assert row["human_citation_supported"] == ""

    row["human_answer_label"] = "correct"
    row["human_citation_supported"] = "true"
    with review.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    assert load_annotations(review, [case])["demo-1"] == {
        "human_answer_label": "correct",
        "human_citation_supported": True,
        "human_notes": "",
    }


def test_annotation_rejects_changed_case_text_and_invalid_labels(tmp_path):
    case = _case()
    review = tmp_path / "review.csv"
    write_annotation_template([case], review)
    with review.open(newline="") as source:
        row = next(csv.DictReader(source))
    row["candidate_answer"] = "Changed after review"
    with review.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="candidate_answer changed"):
        load_annotations(review, [case])


def test_prompt_variants_are_distinct_and_require_separate_decisions():
    case = _case()
    prompts = [build_prompt(name, case) for name in PROMPT_NAMES]
    assert len(set(prompts)) == 3
    assert all("citation_supported" in prompt for prompt in prompts)
    assert "Do not reveal private chain-of-thought" in prompts[-1]


def test_category_summary_reports_answer_and_citation_errors():
    rows = [
        {
            **_case(),
            "human_answer_label": "correct",
            "human_citation_supported": True,
            "naive": {"answer_label": "wrong", "citation_supported": False},
        }
    ]
    report = summarize_prompt(rows, "naive")
    assert report["by_question_type"]["numeric"]["answer_disagreements"] == 1
    assert report["by_question_type"]["numeric"]["citation_disagreements"] == 1


def test_run_reports_costs_and_omits_agreement_until_labeled(monkeypatch):
    case = _case()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "rag_judge_eval.harness._call_judge",
        lambda *args, **kwargs: (
            '{"answer_label":"correct","citation_supported":true,"rationale":"Supported."}',
            {"prompt_tokens": 100, "completion_tokens": 20},
        ),
    )
    report = run_evaluation(
        [case],
        {},
        model="test-model",
        input_cost_per_million=1,
        output_cost_per_million=2,
        human_hourly_rate=60,
        human_minutes_per_case=2,
    )
    assert report["human_labeled_cases"] == 0
    assert report["prompts"]["naive"]["answer_kappa"] is None
    assert report["prompts"]["naive"]["cost_usd"] == pytest.approx(0.00014)
    assert report["prompts"]["naive"]["cost_per_1000_cases_usd"] == pytest.approx(0.14)
    assert report["prompts"]["naive"]["cost_per_1000_label_decisions_usd"] == pytest.approx(0.07)
    assert report["human_label_cost"]["cost_per_1000_labels_usd"] is None
    assert report["prompts"]["rubric"]["usage"]["prompt_tokens"] == 100
