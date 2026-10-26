"""Dataset loading and validation for BenchmarkKit JSONL inputs."""

import json
from pathlib import Path
from typing import Any


def prepare_public_questions(
    path: str | Path, dataset_name: str, limit: int
) -> list[dict[str, str]]:
    """Normalize downloaded HotpotQA or Natural Questions rows to prompt JSONL."""
    if limit < 1:
        raise ValueError("limit must be positive")

    if dataset_name == "hotpotqa":
        try:
            with Path(path).open(encoding="utf-8") as dataset:
                source_records = json.load(dataset)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: expected a HotpotQA JSON array") from error
        if not isinstance(source_records, list):
            raise ValueError(f"{path}: expected a HotpotQA JSON array")
    elif dataset_name == "natural-questions":
        source_records = load_jsonl(path)
    else:
        raise ValueError(f"unsupported public dataset: {dataset_name}")

    prompts = []
    for index, record in enumerate(source_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        question = record.get("question")
        if dataset_name == "natural-questions":
            question = record.get("question_text", question)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}: record {index} has no supported question field")
        identifier = record.get("_id", record.get("example_id", str(index)))
        prompts.append({"prompt": question.strip(), "source": dataset_name, "source_id": str(identifier)})
        if len(prompts) == limit:
            break
    if not prompts:
        raise ValueError(f"{path}: dataset contains no usable questions")
    return prompts


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records and report malformed lines clearly."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset contains no records")
    return records


def validate_prompts(records: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for index, record in enumerate(records, start=1):
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"record {index}: 'prompt' must be a non-empty string")
        prompts.append(prompt)
    return prompts
