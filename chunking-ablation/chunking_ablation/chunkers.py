"""Four chunking strategies with explicit read-time context expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

SENTENCE = re.compile(r"(?<=[.!?])\s+")
STRATEGIES = ("fixed", "recursive", "sentence-window", "semantic")
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    sentence_start: int
    sentence_end: int
    strategy: str
    window_center: int | None = None

    @property
    def token_count(self) -> int:
        return token_count(self.text)


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE.split(text.strip()) if sentence.strip()]


def _pack_sentences(
    document_id: str,
    sentences: list[str],
    groups: list[tuple[int, int]],
    strategy: str,
    target_tokens: int,
    overlap_tokens: int = 0,
) -> list[Chunk]:
    result: list[Chunk] = []
    for start, end in groups:
        current_start = start
        while current_start < end:
            current_end = current_start
            count = 0
            while current_end < end:
                size = token_count(sentences[current_end])
                if current_end > current_start and count + size > target_tokens:
                    break
                count += size
                current_end += 1
            text = " ".join(sentences[current_start:current_end])
            result.append(
                Chunk(
                    f"{document_id}:{strategy}:{len(result):04d}",
                    document_id,
                    text,
                    current_start,
                    current_end,
                    strategy,
                )
            )
            if current_end >= end:
                break
            if overlap_tokens:
                next_start = current_end
                overlap = 0
                while next_start > current_start and overlap < overlap_tokens:
                    next_start -= 1
                    overlap += token_count(sentences[next_start])
                current_start = max(current_start + 1, next_start)
            else:
                current_start = current_end
    return result


def fixed_chunks(
    document_id: str, sentences: list[str], target_tokens: int = 512, overlap_tokens: int = 50
) -> list[Chunk]:
    return _pack_sentences(
        document_id, sentences, [(0, len(sentences))], "fixed", target_tokens, overlap_tokens
    )


def recursive_chunks(
    document_id: str, text: str, target_tokens: int = 512
) -> list[Chunk]:
    sentence_records: list[str] = []
    groups: list[tuple[int, int]] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        start = len(sentence_records)
        sentence_records.extend(_split_sentences(paragraph))
        end = len(sentence_records)
        if end > start:
            groups.append((start, end))
    return _pack_sentences(document_id, sentence_records, groups, "recursive", target_tokens)


def sentence_windows(document_id: str, sentences: list[str]) -> list[Chunk]:
    return [
        Chunk(
            f"{document_id}:sentence-window:{index:04d}",
            document_id,
            sentence,
            index,
            index + 1,
            "sentence-window",
            window_center=index,
        )
        for index, sentence in enumerate(sentences)
    ]


def semantic_chunks(
    document_id: str,
    sentences: list[str],
    vectorizer: TfidfVectorizer,
    threshold: float = 0.12,
    target_tokens: int = 512,
) -> list[Chunk]:
    if not 0 <= threshold <= 1:
        raise ValueError("semantic threshold must be between 0 and 1")
    if not sentences:
        return []
    matrix = vectorizer.transform(sentences)
    groups: list[tuple[int, int]] = []
    start = 0
    count = token_count(sentences[0])
    for index in range(1, len(sentences)):
        similarity = float(matrix[index - 1].multiply(matrix[index]).sum())
        sentence_tokens = token_count(sentences[index])
        if similarity < threshold or count + sentence_tokens > target_tokens:
            groups.append((start, index))
            start = index
            count = sentence_tokens
        else:
            count += sentence_tokens
    groups.append((start, len(sentences)))
    return _pack_sentences(document_id, sentences, groups, "semantic", target_tokens)


def chunk_documents(
    documents: dict[str, str],
    vectorizer: TfidfVectorizer,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 50,
    semantic_threshold: float = 0.12,
) -> dict[str, list[Chunk]]:
    if target_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("target_tokens must be positive and overlap_tokens in [0, target_tokens)")
    sentence_map = {document_id: _split_sentences(text) for document_id, text in documents.items()}
    result: dict[str, list[Chunk]] = {}
    for strategy in STRATEGIES:
        chunks: list[Chunk] = []
        for document_id, text in documents.items():
            sentences = sentence_map[document_id]
            if strategy == "fixed":
                chunks.extend(fixed_chunks(document_id, sentences, target_tokens, overlap_tokens))
            elif strategy == "recursive":
                chunks.extend(recursive_chunks(document_id, text, target_tokens))
            elif strategy == "sentence-window":
                chunks.extend(sentence_windows(document_id, sentences))
            else:
                chunks.extend(
                    semantic_chunks(
                        document_id,
                        sentences,
                        vectorizer,
                        threshold=semantic_threshold,
                        target_tokens=target_tokens,
                    )
                )
        result[strategy] = chunks
    return result


def materialize_context(
    chunk: Chunk,
    sentences_by_document: dict[str, list[str]],
    window_radius: int = 2,
) -> tuple[str, int]:
    if chunk.strategy != "sentence-window":
        return chunk.text, chunk.token_count
    sentences = sentences_by_document[chunk.document_id]
    center = chunk.window_center
    if center is None:
        raise ValueError("sentence-window chunk is missing its center sentence")
    start = max(0, center - window_radius)
    end = min(len(sentences), center + window_radius + 1)
    context = " ".join(sentences[start:end])
    return context, token_count(context)
