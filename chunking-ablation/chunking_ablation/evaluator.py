"""A fixed TF-IDF retrieval baseline, extractive answers, and optional LLM calls."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking_ablation.chunkers import Chunk, materialize_context


def fit_embedder(documents: dict[str, str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(list(documents.values()))
    return vectorizer


def retrieve(
    question: str,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer,
    sentences_by_document: dict[str, list[str]],
    k: int = 5,
    window_radius: int = 2,
    chunk_vectors: Any = None,
) -> list[tuple[Chunk, str, int, float]]:
    if k < 1:
        raise ValueError("k must be positive")
    if not chunks:
        return []
    chunk_matrix = chunk_vectors
    if chunk_matrix is None:
        chunk_matrix = vectorizer.transform([chunk.text for chunk in chunks])
    query = vectorizer.transform([question])
    scores = (chunk_matrix @ query.T).toarray().ravel()
    order = np.argsort(-scores, kind="stable")[:k]
    result = []
    for index in order:
        chunk = chunks[int(index)]
        context, context_tokens = materialize_context(
            chunk, sentences_by_document, window_radius
        )
        result.append((chunk, context, context_tokens, float(scores[index])))
    return result


def extractive_answer(question: str, contexts: list[str]) -> str:
    question_terms = set(question.casefold().split())
    candidates = []
    for context in contexts:
        for sentence in context.split(". "):
            normalized = sentence.strip()
            if normalized and not normalized.endswith((".", "!", "?")):
                normalized += "."
            terms = set(normalized.casefold().split())
            overlap = len(question_terms & terms)
            candidates.append((overlap, normalized))
    return max(candidates, key=lambda item: item[0])[1] if candidates else ""


def _chat_completion(
    messages: list[dict[str, str]], *, model: str, max_tokens: int
) -> tuple[str, dict[str, int]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --llm is enabled")
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"chat completion request failed: {error}") from error
    try:
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return text.strip(), {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("chat completion returned an invalid response shape") from error


def answer_question(
    question: str,
    contexts: list[str],
    *,
    use_llm: bool,
    model: str,
) -> tuple[str, dict[str, int]]:
    if not use_llm:
        return extractive_answer(question, contexts), {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = (
        "Answer the question using only the provided context. If the answer is not supported, "
        "say 'Not enough evidence'. Return a concise answer.\n\n"
        f"Question: {question}\n\nContext:\n" + "\n\n".join(contexts)
    )
    return _chat_completion(
        [{"role": "user", "content": prompt}], model=model, max_tokens=160
    )


def judge_answer(
    question: str,
    answer: str,
    expected_answer: str,
    evidence: str,
    *,
    model: str,
) -> tuple[bool, dict[str, int]]:
    prompt = (
        "Judge whether the candidate answer is factually equivalent to the reference answer "
        "and supported by the evidence. Reply with only JSON: {\"correct\": true or false}.\n\n"
        f"Question: {question}\nReference answer: {expected_answer}\nCandidate answer: {answer}\n"
        f"Evidence: {evidence}"
    )
    text, usage = _chat_completion(
        [{"role": "user", "content": prompt}], model=model, max_tokens=30
    )
    try:
        result: Any = json.loads(text)
        correct = result["correct"]
        if not isinstance(correct, bool):
            raise ValueError
        return correct, usage
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("LLM judge must return JSON with a boolean 'correct' field") from error
