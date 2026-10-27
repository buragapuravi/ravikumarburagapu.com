"""Skip-gram with negative sampling, implemented with NumPy."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass
class SkipGramNS:
    words: list[str]
    word_to_id: dict[str, int]
    input_vectors: np.ndarray
    output_vectors: np.ndarray
    token_counts: np.ndarray
    epochs: int
    elapsed_seconds: float
    losses: list[float]

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        dimension: int = 100,
        window: int = 5,
        negatives: int = 5,
        epochs: int = 5,
        min_count: int = 1,
        learning_rate: float = 0.025,
        batch_size: int = 256,
        seed: int = 17,
    ) -> "SkipGramNS":
        if dimension < 1 or window < 1 or negatives < 1 or epochs < 1:
            raise ValueError("dimension, window, negatives, and epochs must be positive")
        if min_count < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("min_count, batch_size, and learning_rate must be positive")

        start = time.perf_counter()
        sentences = [tokenize(text) for text in texts]
        from collections import Counter

        counts = Counter(token for sentence in sentences for token in sentence)
        words = sorted(word for word, count in counts.items() if count >= min_count)
        if len(words) < 2:
            raise ValueError("training corpus must contain at least two retained tokens")
        word_to_id = {word: index for index, word in enumerate(words)}
        token_counts = np.asarray([counts[word] for word in words], dtype=np.float64)
        encoded = [
            [word_to_id[token] for token in sentence if token in word_to_id]
            for sentence in sentences
        ]
        centers: list[int] = []
        targets: list[int] = []
        for sentence in encoded:
            for position, center in enumerate(sentence):
                left = max(0, position - window)
                right = min(len(sentence), position + window + 1)
                for context_position in range(left, right):
                    if context_position != position:
                        centers.append(center)
                        targets.append(sentence[context_position])
        if not centers:
            raise ValueError("corpus has no skip-gram context pairs; add longer sentences")

        center_ids = np.asarray(centers, dtype=np.int64)
        target_ids = np.asarray(targets, dtype=np.int64)
        rng = np.random.default_rng(seed)
        scale = 0.5 / dimension
        input_vectors = rng.uniform(-scale, scale, (len(words), dimension)).astype(np.float32)
        output_vectors = np.zeros((len(words), dimension), dtype=np.float32)
        noise = token_counts**0.75
        noise /= noise.sum()
        losses = []
        total_steps = epochs * ((len(center_ids) + batch_size - 1) // batch_size)
        step = 0

        for _epoch in range(epochs):
            order = rng.permutation(len(center_ids))
            epoch_loss = 0.0
            seen = 0
            for offset in range(0, len(order), batch_size):
                batch = order[offset : offset + batch_size]
                centers_batch = center_ids[batch]
                targets_batch = target_ids[batch]
                size = len(batch)
                negative_ids = rng.choice(
                    len(words), size=(size, negatives), replace=True, p=noise
                )
                collisions = negative_ids == targets_batch[:, None]
                while collisions.any():
                    negative_ids[collisions] = rng.choice(
                        len(words), size=int(collisions.sum()), replace=True, p=noise
                    )
                    collisions = negative_ids == targets_batch[:, None]
                positive_vectors = output_vectors[targets_batch].copy()
                negative_vectors = output_vectors[negative_ids].copy()
                center_vectors = input_vectors[centers_batch].copy()

                positive_logits = np.einsum("bd,bd->b", center_vectors, positive_vectors)
                negative_logits = np.einsum("bd,bkd->bk", center_vectors, negative_vectors)
                positive_probability = _sigmoid(positive_logits)
                negative_probability = _sigmoid(negative_logits)
                grad_positive = positive_probability - 1.0
                grad_negative = negative_probability
                grad_centers = (
                    grad_positive[:, None] * positive_vectors
                    + np.einsum("bk,bkd->bd", grad_negative, negative_vectors)
                )

                learning_rate_now = learning_rate * max(
                    0.0001, 1.0 - step / max(total_steps, 1)
                )
                np.add.at(
                    input_vectors,
                    centers_batch,
                    -learning_rate_now * grad_centers,
                )
                np.add.at(
                    output_vectors,
                    targets_batch,
                    -learning_rate_now * grad_positive[:, None] * center_vectors,
                )
                np.add.at(
                    output_vectors,
                    negative_ids.ravel(),
                    (
                        -learning_rate_now
                        * grad_negative[:, :, None]
                        * center_vectors[:, None, :]
                    ).reshape(-1, dimension),
                )

                epoch_loss += float(
                    np.logaddexp(0.0, -positive_logits).sum()
                    + np.logaddexp(0.0, negative_logits).sum()
                )
                seen += size
                step += 1
            losses.append(epoch_loss / max(seen, 1))

        return cls(
            words=words,
            word_to_id=word_to_id,
            input_vectors=input_vectors,
            output_vectors=output_vectors,
            token_counts=token_counts.astype(np.int64),
            epochs=epochs,
            elapsed_seconds=time.perf_counter() - start,
            losses=losses,
        )

    @property
    def vectors(self) -> np.ndarray:
        return self.input_vectors


@dataclass
class EmbeddingSpace:
    words: list[str]
    word_to_id: dict[str, int]
    vectors: np.ndarray
    token_counts: np.ndarray
    elapsed_seconds: float
    algorithm: str


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # Clipping avoids overflow for very large intermediate logits.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
