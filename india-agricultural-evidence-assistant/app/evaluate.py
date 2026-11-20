from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from math import log2
from pathlib import Path
from typing import Any

from app.answer import split_answer_sentences
from app.config import ROOT, settings
from app.engine import ask
from app.ingest import create_store
from app.retrieval import evidence_gate, retrieve

_RENDERED_CITATION = re.compile(r"\[[^\]]+,\s*(?:p\.\s*\d+|page unavailable)\]")
_TOKEN_CITATION = re.compile(r"\[([a-zA-Z0-9._-]+)\|(p\.\s*\d+|n\.p\.)\]")


def _citation_checks(result) -> dict[str, Any]:
    sentences = [
        sentence
        for sentence in split_answer_sentences(result.answer)
        if sentence.casefold().strip(" .!?") not in {"yes", "no"}
    ]
    covered = sum(
        bool(_RENDERED_CITATION.search(sentence) or _TOKEN_CITATION.search(sentence))
        for sentence in sentences
    )
    evidence_pages = {
        (item.chunk.metadata.source_id, item.chunk.page_number)
        for item in result.evidence
    }
    valid = 0
    for token in result.citations:
        match = _TOKEN_CITATION.fullmatch(token)
        if not match:
            continue
        page = int(re.search(r"\d+", match.group(2)).group()) if match.group(2) != "n.p." else None
        valid += (match.group(1), page) in evidence_pages
    return {
        "citation_coverage": 1.0 if result.abstained else covered / max(1, len(sentences)),
        "valid_citations": valid,
        "citation_validity": valid / max(1, len(result.citations)),
    }


def _ragas_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    provider = os.getenv("RAGAS_LLM_PROVIDER", "none").lower()
    if provider == "none":
        return {"status": "skipped", "reason": "RAGAS_LLM_PROVIDER is none; deterministic checks still ran."}
    if provider != "openai-compatible":
        return {"status": "skipped", "reason": "Ragas diagnostics support the OpenAI-compatible provider only."}
    try:
        from openai import APIError
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as error:
        return {"status": "unavailable", "reason": f"Install the evaluation extra: {error}"}

    api_key = os.getenv("RAGAS_API_KEY", "")
    base_url = os.getenv("RAGAS_BASE_URL", "")
    model_name = os.getenv("RAGAS_MODEL", "")
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "")
    if not model_name or not base_url or not embedding_model:
        return {
            "status": "unavailable",
            "reason": "Set RAGAS_MODEL, RAGAS_EMBEDDING_MODEL, and RAGAS_BASE_URL in .env.",
        }
    try:
        timeout = float(os.getenv("RAGAS_TIMEOUT_SECONDS", "45"))
        if timeout <= 0:
            return {"status": "unavailable", "reason": "RAGAS_TIMEOUT_SECONDS must be positive."}
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key or "not-required",
                timeout=timeout,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=embedding_model,
            base_url=base_url,
            api_key=api_key or "not-required",
            request_timeout=timeout,
        ))
        dataset = EvaluationDataset.from_list(rows)
        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(llm=llm),
                ResponseRelevancy(llm=llm, embeddings=embeddings),
                LLMContextPrecisionWithReference(llm=llm),
                LLMContextRecall(llm=llm),
            ],
        )
    except (APIError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        return {"status": "error", "reason": str(error)}
    scores = json.loads(result.to_pandas().to_json(orient="records"))
    return {"status": "completed", "scores": scores}


def _calibrate_thresholds(
    examples: list[dict[str, Any]],
    max_source_age_days: int,
) -> dict[str, Any]:
    scores = (0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35)
    coverages = (0.0, 0.05, 0.1, 0.15, 0.25, 0.4)
    candidates = []
    positives = sum(bool(item["expected"]) for item in examples)
    negatives = len(examples) - positives
    for minimum_score in scores:
        for minimum_coverage in coverages:
            tp = tn = 0
            for item in examples:
                decision = evidence_gate(
                    item["question"],
                    item["evidence"],
                    minimum_score,
                    minimum_coverage,
                    max_official_source_age_days=max_source_age_days,
                )
                if item["expected"] and decision.answerable:
                    tp += 1
                if not item["expected"] and not decision.answerable:
                    tn += 1
            balanced_accuracy = 0.5 * (
                tp / max(1, positives) + tn / max(1, negatives)
            )
            candidates.append(
                {
                    "minimum_score": minimum_score,
                    "minimum_coverage": minimum_coverage,
                    "balanced_accuracy": balanced_accuracy,
                    "true_positive": tp,
                    "true_negative": tn,
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["balanced_accuracy"],
            item["minimum_score"],
            item["minimum_coverage"],
        ),
    )
    return {
        "method": "Grid search maximizing balanced accuracy on the bundled curated examples.",
        "selected": selected,
        "candidates": candidates,
        "warning": "Tiny synthetic teaching set; these thresholds are not agronomic safety calibration.",
    }


def _retrieval_metrics(
    ranked_source_ids: list[str],
    relevant_source_ids: list[str],
    k: int,
) -> dict[str, float]:
    relevant = set(relevant_source_ids)
    if not relevant:
        raise ValueError("Retrieval benchmark examples must define relevant_source_ids.")
    top_k = ranked_source_ids[:k]
    relevant_ranks = [
        rank
        for rank, source_id in enumerate(ranked_source_ids, start=1)
        if source_id in relevant
    ]
    dcg = sum(
        1 / log2(rank + 1)
        for rank, source_id in enumerate(top_k, start=1)
        if source_id in relevant
    )
    ideal_dcg = sum(
        1 / log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return {
        f"precision@{k}": sum(source_id in relevant for source_id in top_k) / k,
        f"recall@{k}": sum(source_id in relevant for source_id in top_k) / len(relevant),
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _evaluate_retrieval(
    dataset_path: Path,
    config,
    k: int = 5,
) -> dict[str, Any]:
    examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("The retrieval benchmark must be a non-empty JSON list.")
    store = create_store(config)
    per_query = []
    for example in examples:
        relevant = example.get("relevant_source_ids", [])
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Retrieval benchmark question must have relevant_source_ids: {example.get('question', '')}"
            )
        ranked_chunks, _ = retrieve(
            store=store,
            query=example["question"],
            framework=config.framework,
            dense_limit=config.dense_top_k,
            lexical_limit=config.lexical_top_k,
            rerank_limit=config.rerank_top_k,
            reranker_model=config.reranker_model,
        )
        ranked_source_ids = list(
            dict.fromkeys(item.chunk.metadata.source_id for item in ranked_chunks)
        )
        metrics = _retrieval_metrics(ranked_source_ids, relevant, k)
        per_query.append(
            {
                "question": example["question"],
                "relevant_source_ids": relevant,
                "ranked_source_ids": ranked_source_ids,
                "relevant_source_ranks": [
                    rank
                    for rank, source_id in enumerate(ranked_source_ids, start=1)
                    if source_id in set(relevant)
                ],
                **metrics,
            }
        )
    metric_names = [f"precision@{k}", f"recall@{k}", "reciprocal_rank", f"ndcg@{k}"]
    aggregate = {
        name: sum(row[name] for row in per_query) / len(per_query)
        for name in metric_names
    }
    return {
        "dataset": str(dataset_path),
        "k": k,
        "query_count": len(per_query),
        "source_level_metrics": "Each source is counted once, at its first-ranked retrieved chunk.",
        "aggregate": aggregate,
        "examples": per_query,
    }


def evaluate_dataset(
    dataset_path: Path,
    report_dir: Path,
    retrieval_dataset_path: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    config = settings()
    retrieval_dataset_path = retrieval_dataset_path or ROOT / "data/evaluation/retrieval_questions.json"
    examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    result_rows: list[dict[str, Any]] = []
    true_positive = true_negative = false_positive = false_negative = 0
    judge_rows: list[dict[str, Any]] = []
    calibration_examples: list[dict[str, Any]] = []
    for example in examples:
        answer = ask(example["question"], config)
        expected = bool(example["expected_answerability"])
        actual = not answer.abstained
        calibration_examples.append(
            {
                "question": example["question"],
                "expected": expected,
                "evidence": answer.evidence,
            }
        )
        if expected and actual:
            true_positive += 1
        elif not expected and not actual:
            true_negative += 1
        elif not expected and actual:
            false_positive += 1
        else:
            false_negative += 1
        citation = _citation_checks(answer)
        valid_reference = all(
            source_id in {item.chunk.metadata.source_id for item in answer.evidence}
            for source_id in answer.source_ids
        )
        row = {
            "question": example["question"],
            "expected_answerability": expected,
            "actual_answerability": actual,
            "abstained": answer.abstained,
            "abstention_reason": answer.abstention_reason,
            "answer": answer.answer,
            "expected_source_ids": example["reference_source_ids"],
            "returned_source_ids": answer.source_ids,
            "citation_reference_valid": valid_reference,
            **citation,
        }
        result_rows.append(row)
        if expected and example.get("reference_answer"):
            judge_rows.append(
                {
                    "user_input": example["question"],
                    "response": answer.answer,
                    "retrieved_contexts": [item.chunk.text for item in answer.evidence],
                    "reference": example["reference_answer"],
                }
            )

    confusion = {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "abstention_precision": true_negative / max(1, true_negative + false_positive),
        "abstention_recall": true_negative / max(1, true_negative + false_negative),
    }
    ragas = _ragas_diagnostics(judge_rows)
    answerable_rows = [row for row in result_rows if row["actual_answerability"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "model_judged_metrics_note": (
            "Ragas results are diagnostic, apply only to answerable items, and require human spot checks; "
            "they are not proof of safety or correctness."
        ),
        "confusion_matrix": confusion,
        "threshold_calibration": _calibrate_thresholds(
            calibration_examples,
            config.max_official_source_age_days,
        ),
        "retrieval_benchmark": _evaluate_retrieval(retrieval_dataset_path, config),
        "deterministic_checks": {
            "mean_citation_coverage": sum(
                row["citation_coverage"] for row in answerable_rows
            ) / max(1, len(answerable_rows)),
            "citation_coverage_population": "Answerable responses only; abstentions contain no factual answer claims.",
            "all_returned_source_ids_valid": all(row["citation_reference_valid"] for row in result_rows),
            "unanswerable_abstentions": sum(
                row["abstained"] for row in result_rows if not row["expected_answerability"]
            ),
        },
        "ragas": ragas,
        "examples": result_rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = report_dir / f"evaluation-{timestamp}.json"
    csv_path = report_dir / f"evaluation-{timestamp}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    return json_path, csv_path, report


def main() -> None:
    dataset = ROOT / "data/evaluation/questions.json"
    report_dir = ROOT / "reports"
    json_path, csv_path, report = evaluate_dataset(dataset, report_dir)
    print(json.dumps({"json_report": str(json_path), "csv_report": str(csv_path), **report["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
