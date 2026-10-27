"""Comparable GloVe and optional FastText baselines."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from word2vec_lab.model import EmbeddingSpace, tokenize


def train_glove(
    texts: Iterable[str],
    *,
    dimension: int,
    window: int = 5,
    epochs: int = 15,
    min_count: int = 1,
    learning_rate: float = 0.05,
    seed: int = 17,
) -> EmbeddingSpace:
    """Train a compact GloVe-style model using sparse co-occurrence entries."""
    start = time.perf_counter()
    sentences = [tokenize(text) for text in texts]
    counts = Counter(token for sentence in sentences for token in sentence)
    words = sorted(word for word, count in counts.items() if count >= min_count)
    if len(words) < 2:
        raise ValueError("training corpus must contain at least two retained tokens")
    word_to_id = {word: index for index, word in enumerate(words)}
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)
    for sentence in sentences:
        encoded = [word_to_id[word] for word in sentence if word in word_to_id]
        for position, center in enumerate(encoded):
            for other_position in range(
                max(0, position - window), min(len(encoded), position + window + 1)
            ):
                if other_position != position:
                    distance = abs(position - other_position)
                    cooccurrence[(center, encoded[other_position])] += 1.0 / distance
    if not cooccurrence:
        raise ValueError("corpus has no co-occurrences; add longer sentences")

    rng = np.random.default_rng(seed)
    scale = 0.5 / dimension
    word_vectors = rng.uniform(-scale, scale, (len(words), dimension))
    context_vectors = rng.uniform(-scale, scale, (len(words), dimension))
    word_bias = np.zeros(len(words))
    context_bias = np.zeros(len(words))
    accum_word = np.zeros_like(word_vectors)
    accum_context = np.zeros_like(context_vectors)
    accum_word_bias = np.zeros_like(word_bias)
    accum_context_bias = np.zeros_like(context_bias)
    entries = np.asarray(
        [(i, j, count) for (i, j), count in cooccurrence.items()],
        dtype=np.float64,
    )
    weight_max, alpha = 100.0, 0.75
    for _ in range(epochs):
        rng.shuffle(entries)
        for center, context, count in entries:
            i, j = int(center), int(context)
            weight = min(1.0, (count / weight_max) ** alpha)
            difference = (
                np.dot(word_vectors[i], context_vectors[j])
                + word_bias[i]
                + context_bias[j]
                - np.log(count)
            )
            gradient = 2.0 * weight * difference
            grad_word = gradient * context_vectors[j]
            grad_context = gradient * word_vectors[i]
            accum_word[i] += grad_word**2
            accum_context[j] += grad_context**2
            accum_word_bias[i] += gradient**2
            accum_context_bias[j] += gradient**2
            word_vectors[i] -= learning_rate * grad_word / np.sqrt(
                np.maximum(accum_word[i], 1e-12)
            )
            context_vectors[j] -= learning_rate * grad_context / np.sqrt(
                np.maximum(accum_context[j], 1e-12)
            )
            word_bias[i] -= learning_rate * gradient / np.sqrt(
                max(accum_word_bias[i], 1e-12)
            )
            context_bias[j] -= learning_rate * gradient / np.sqrt(
                max(accum_context_bias[j], 1e-12)
            )
    vectors = word_vectors + context_vectors
    return EmbeddingSpace(
        words=words,
        word_to_id=word_to_id,
        vectors=vectors.astype(np.float32),
        token_counts=np.asarray([counts[word] for word in words], dtype=np.int64),
        elapsed_seconds=time.perf_counter() - start,
        algorithm="glove",
    )


def train_fasttext(
    texts: Iterable[str],
    *,
    dimension: int,
    window: int = 5,
    epochs: int = 5,
    min_count: int = 1,
    seed: int = 17,
) -> EmbeddingSpace:
    try:
        from gensim.models import FastText
    except ImportError as error:
        raise RuntimeError(
            "FastText baseline requires the optional dependency; install with "
            "`pip install -e '.[fasttext]'`"
        ) from error

    start = time.perf_counter()
    sentences = [tokenize(text) for text in texts]
    model = FastText(
        sentences=sentences,
        vector_size=dimension,
        window=window,
        min_count=min_count,
        sg=1,
        negative=5,
        workers=1,
        seed=seed,
        epochs=epochs,
        sorted_vocab=1,
    )
    words = list(model.wv.index_to_key)
    vectors = np.asarray([model.wv[word] for word in words], dtype=np.float32)
    counts = np.asarray([model.wv.get_vecattr(word, "count") for word in words])
    return EmbeddingSpace(
        words=words,
        word_to_id={word: index for index, word in enumerate(words)},
        vectors=vectors,
        token_counts=counts,
        elapsed_seconds=time.perf_counter() - start,
        algorithm="fasttext",
    )
