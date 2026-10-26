"""The three judge prompt variants evaluated by the harness."""

from __future__ import annotations

import json
from typing import Any

PROMPT_NAMES = ("naive", "rubric", "chain_of_thought")


def build_prompt(name: str, case: dict[str, Any]) -> str:
    if name not in PROMPT_NAMES:
        raise ValueError(f"unknown prompt variant: {name}")

    shared = (
        f"Question: {case['question']}\n"
        f"Reference answer: {case['reference_answer']}\n"
        f"Candidate answer: {case['candidate_answer']}\n"
        f"Cited passages:\n{json.dumps(case['cited_passages'], ensure_ascii=False)}"
    )
    instructions = {
        "naive": (
            "Judge this RAG answer. Is it correct, partially correct, or wrong compared "
            "with the reference? Do the cited passages support the candidate answer?"
        ),
        "rubric": (
            "Evaluate answer correctness and citation support independently.\n"
            "correct: all material claims answer the question and agree with the reference.\n"
            "partially_correct: a material part is right, but the answer is incomplete or "
            "contains a minor error.\n"
            "wrong: the answer is materially incorrect, irrelevant, or unsupported by the "
            "reference.\n"
            "citation_supported is true only when the cited passages directly support the "
            "candidate's material claims. Do not infer support from the reference answer."
        ),
        "chain_of_thought": (
            "Assess the answer in stages: identify its material claims; compare them with "
            "the reference; then check whether the cited passages support those claims. "
            "Classify the answer as correct, partially_correct, or wrong, and independently "
            "decide citation_supported. Do not reveal private chain-of-thought; return only "
            "the requested labels and a concise evidence-based rationale."
        ),
    }[name]
    return (
        f"{instructions}\n\n{shared}\n\n"
        "Return only a JSON object with this exact shape: "
        '{"answer_label":"correct|partially_correct|wrong",'
        '"citation_supported":true,"rationale":"brief explanation"}'
    )
