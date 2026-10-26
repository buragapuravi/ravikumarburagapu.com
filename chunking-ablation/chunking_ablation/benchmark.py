"""Run identical retrieval/QA fixtures across chunking strategies."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking_ablation.chunkers import (
    SENTENCE,
    STRATEGIES,
    chunk_documents,
)
from chunking_ablation.evaluator import answer_question, fit_embedder, judge_answer, retrieve
from chunking_ablation.metrics import (
    context_token_cost,
    exact_answer_accuracy,
    recall_at_k,
    required_phrase_coverage,
)


def load_documents(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise ValueError(f"corpus directory does not exist: {directory}")
    documents = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.md"))
        if path.is_file() and not path.is_symlink()
    }
    if not documents:
        raise ValueError(f"no Markdown corpus files found in {directory}")
    return documents


def load_questions(path: Path) -> list[dict[str, str]]:
    questions = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            required = {"id", "document_id", "question", "expected_answer", "evidence"}
            if not required.issubset(row) or not all(
                isinstance(row[key], str) and row[key].strip() for key in required
            ):
                raise ValueError("missing or empty required fields")
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid Q/A JSONL at line {line_number}: {error}") from error
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate Q/A id at line {line_number}: {row['id']}")
        seen_ids.add(row["id"])
        questions.append(row)
    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions


def _sentences_by_document(documents: dict[str, str]) -> dict[str, list[str]]:
    return {
        doc_id: [sentence.strip() for sentence in SENTENCE.split(text.strip()) if sentence.strip()]
        for doc_id, text in documents.items()
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def run_benchmark(
    documents: dict[str, str],
    questions: list[dict[str, str]],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 50,
    semantic_threshold: float = 0.12,
    k: int = 5,
    window_radius: int = 2,
    llm: bool = False,
    judge: bool = False,
    model: str = "gpt-4o-mini",
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
) -> dict[str, Any]:
    if k != 5:
        raise ValueError("this benchmark fixes retrieval depth at k=5")
    if llm or judge:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when --llm or --judge is enabled")
    missing_docs = {row.get("document_id") for row in questions} - set(documents)
    if missing_docs:
        raise ValueError(f"Q/A fixture references missing documents: {sorted(missing_docs)}")
    vectorizer = fit_embedder(documents)
    sentences = _sentences_by_document(documents)
    chunks_by_strategy = chunk_documents(
        documents,
        vectorizer,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        semantic_threshold=semantic_threshold,
    )
    output_runs = []
    for strategy in STRATEGIES:
        strategy_chunks = chunks_by_strategy[strategy]
        chunk_vectors = vectorizer.transform([chunk.text for chunk in strategy_chunks])
        per_query = []
        retrieval_latencies = []
        end_to_end_latencies = []
        context_pairs: list[tuple[str, int]] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        judged_scores: list[float] = []
        for row in questions:
            started = time.perf_counter()
            retrieval_started = time.perf_counter()
            retrieved = retrieve(
                row["question"],
                strategy_chunks,
                vectorizer,
                sentences,
                k=k,
                window_radius=window_radius,
                chunk_vectors=chunk_vectors,
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            contexts = [item[1] for item in retrieved]
            context_tokens = sum(item[2] for item in retrieved)
            answer, usage = answer_question(
                row["question"], contexts, use_llm=llm, model=model
            )
            prompt_tokens_total += usage["prompt_tokens"]
            completion_tokens_total += usage["completion_tokens"]
            exact = exact_answer_accuracy(answer, row["expected_answer"])
            coverage = required_phrase_coverage(answer, row["expected_answer"])
            judged_correct: bool | None = None
            judge_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            if judge:
                judged_correct, judge_usage = judge_answer(
                    row["question"],
                    answer,
                    row["expected_answer"],
                    row["evidence"],
                    model=model,
                )
                prompt_tokens_total += judge_usage["prompt_tokens"]
                completion_tokens_total += judge_usage["completion_tokens"]
                judged_scores.append(float(judged_correct))
            elapsed = time.perf_counter() - started
            retrieval_latencies.append(retrieval_seconds * 1000)
            end_to_end_latencies.append(elapsed * 1000)
            context_pairs.append((row["question"], context_tokens))
            per_query.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "evidence": row["evidence"],
                    "answer": answer,
                    "retrieved_chunk_ids": [item[0].chunk_id for item in retrieved],
                    "retrieved_contexts": contexts,
                    "retrieval_recall_at_5": recall_at_k(
                        [item[0] for item in retrieved], row["evidence"], k=5
                    ),
                    "exact_match": exact,
                    "reference_phrase_coverage": coverage,
                    "llm_judge_correct": judged_correct,
                    "human_review": "pending",
                    "context_tokens": context_tokens,
                    "retrieval_latency_ms": retrieval_seconds * 1000,
                    "end_to_end_latency_ms": elapsed * 1000,
                }
            )
        retrieval_recall = float(np.mean([row["retrieval_recall_at_5"] for row in per_query]))
        exact_accuracy = float(np.mean([row["exact_match"] for row in per_query]))
        cost_tokens = prompt_tokens_total if llm or judge else 0
        avg_cost = (
            cost_tokens * input_cost_per_million
            + completion_tokens_total * output_cost_per_million
        ) / 1_000_000 / len(questions)
        output_runs.append(
            {
                "strategy": strategy,
                "chunk_count": len(strategy_chunks),
                "mean_chunk_tokens": float(np.mean([chunk.token_count for chunk in strategy_chunks])),
                "retrieval_recall_at_5": retrieval_recall,
                "exact_answer_accuracy": exact_accuracy,
                "llm_judge_accuracy": float(np.mean(judged_scores)) if judged_scores else None,
                "mean_reference_phrase_coverage": float(np.mean([row["reference_phrase_coverage"] for row in per_query])),
                "avg_context_tokens": float(np.mean([tokens for _, tokens in context_pairs])),
                "avg_cost_per_query": avg_cost,
                "input_cost_per_million_tokens": input_cost_per_million,
                "output_cost_per_million_tokens": output_cost_per_million,
                "mean_retrieval_latency_ms": float(np.mean(retrieval_latencies)),
                "p95_retrieval_latency_ms": _percentile(retrieval_latencies, 95),
                "mean_end_to_end_latency_ms": float(np.mean(end_to_end_latencies)),
                "p95_end_to_end_latency_ms": _percentile(end_to_end_latencies, 95),
                "total_prompt_tokens": prompt_tokens_total,
                "total_completion_tokens": completion_tokens_total,
                "queries": per_query,
            }
        )
    return {
        "dataset": {
            "document_count": len(documents),
            "question_count": len(questions),
            "corpus": "original illustrative engineering notes",
            "qa_fixture": "50 hand-authored questions with evidence sentences",
        },
        "configuration": {
            "embedder": "scikit-learn TF-IDF (word and bigram; fitted once, shared by all chunkers)",
            "context_tokenizer": "tiktoken cl100k_base",
            "answerer": f"OpenAI-compatible {model}" if llm else "deterministic extractive baseline",
            "judge": f"OpenAI-compatible {model}" if judge else None,
            "k": 5,
            "target_chunk_tokens": target_tokens,
            "fixed_overlap_tokens": overlap_tokens,
            "sentence_window_radius": window_radius,
            "semantic_adjacent_cosine_threshold": semantic_threshold,
            "llm_judge_validation": "LLM judge scores are estimates; human review is pending for every example.",
        },
        "runs": output_runs,
    }
