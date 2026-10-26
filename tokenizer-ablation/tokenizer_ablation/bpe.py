"""A small byte-level BPE trainer and encoder implemented with the Python stdlib."""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

_PIECES = re.compile(r"\s+|[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]")


@dataclass(frozen=True)
class BytePairTokenizer:
    """Byte-fallback tokenizer; merge order is the learned BPE model."""

    merges: tuple[tuple[int, int], ...]
    vocab_size: int

    def __post_init__(self) -> None:
        if self.vocab_size < 256 or self.vocab_size > 256 + len(self.merges):
            raise ValueError("vocab_size must be between 256 and the learned vocabulary size")

    @cached_property
    def _merge_ranks(self) -> dict[tuple[int, int], tuple[int, int]]:
        return {
            pair: (rank + 256, rank)
            for rank, pair in enumerate(self.merges[: self.vocab_size - 256])
        }

    def encode(self, text: str) -> list[int]:
        ranks = self._merge_ranks
        output: list[int] = []
        for piece in _PIECES.findall(text):
            tokens = list(piece.encode("utf-8"))
            while len(tokens) > 1:
                best_rank: int | None = None
                best_pair: tuple[int, int] | None = None
                for left, right in zip(tokens, tokens[1:]):
                    merge = ranks.get((left, right))
                    if merge is not None and (best_rank is None or merge[1] < best_rank):
                        best_pair = (left, right)
                        best_rank = merge[1]
                if best_pair is None:
                    break
                merged: list[int] = []
                position = 0
                while position < len(tokens):
                    if (
                        position + 1 < len(tokens)
                        and (tokens[position], tokens[position + 1]) == best_pair
                    ):
                        merged.append(ranks[best_pair][0])
                        position += 2
                    else:
                        merged.append(tokens[position])
                        position += 1
                tokens = merged
            output.extend(tokens)
        return output


def train_bpe(texts: Iterable[str], vocab_size: int) -> BytePairTokenizer:
    """Train deterministic byte BPE, maintaining pair counts as words are merged."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 for byte fallback")
    frequencies: Counter[str] = Counter()
    for text in texts:
        frequencies.update(_PIECES.findall(text))
    if not frequencies:
        raise ValueError("training corpus must contain text")

    sequences = [list(piece.encode("utf-8")) for piece in frequencies]
    weights = [frequencies[piece] for piece in frequencies]
    word_pairs: list[dict[tuple[int, int], int]] = []
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_counts: Counter[tuple[int, int]] = Counter()

    def count_pairs(sequence: list[int]) -> Counter[tuple[int, int]]:
        return Counter(zip(sequence, sequence[1:]))

    for word_id, sequence in enumerate(sequences):
        counts = count_pairs(sequence)
        word_pairs.append(dict(counts))
        for pair, count in counts.items():
            pair_counts[pair] += count * weights[word_id]
            pair_words[pair].add(word_id)

    heap = [(-count, pair[0], pair[1]) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges: list[tuple[int, int]] = []

    def update_pair(pair: tuple[int, int], delta: int, word_id: int) -> None:
        pair_counts[pair] += delta
        if word_pairs[word_id].get(pair, 0) == 0:
            pair_words[pair].discard(word_id)
        elif delta > 0:
            pair_words[pair].add(word_id)
        if pair_counts[pair] <= 0:
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
        else:
            heapq.heappush(heap, (-pair_counts[pair], pair[0], pair[1]))

    while len(merges) < vocab_size - 256:
        while heap:
            negative_count, left, right = heapq.heappop(heap)
            pair = (left, right)
            if pair_counts.get(pair) == -negative_count:
                break
        else:
            break
        new_token = 256 + len(merges)
        merges.append(pair)
        affected = tuple(pair_words.get(pair, ()))
        for word_id in affected:
            sequence = sequences[word_id]
            old_counts = word_pairs[word_id]
            word_pairs[word_id] = {}
            for old_pair, count in old_counts.items():
                update_pair(old_pair, -count * weights[word_id], word_id)
            merged: list[int] = []
            position = 0
            while position < len(sequence):
                if position + 1 < len(sequence) and sequence[position : position + 2] == list(pair):
                    merged.append(new_token)
                    position += 2
                else:
                    merged.append(sequence[position])
                    position += 1
            sequences[word_id] = merged
            new_counts = count_pairs(merged)
            word_pairs[word_id] = dict(new_counts)
            for new_pair, count in new_counts.items():
                update_pair(new_pair, count * weights[word_id], word_id)
    return BytePairTokenizer(tuple(merges), 256 + len(merges))
