import numpy as np
import pytest

from word2vec_lab.evaluation import evaluate_analogies, evaluate_neighbors, nearest_neighbors
from word2vec_lab.model import EmbeddingSpace, SkipGramNS, tokenize


def test_tokenizer_is_case_insensitive_and_keeps_apostrophes():
    assert tokenize("Don't repeat WORDS; words!") == ["don't", "repeat", "words", "words"]


def test_skipgram_training_is_seeded_and_returns_finite_vectors():
    corpus = ["red apple grows", "green apple falls", "red berry grows"]
    first = SkipGramNS.train(
        corpus, dimension=8, window=2, negatives=2, epochs=2, batch_size=4, seed=3
    )
    second = SkipGramNS.train(
        corpus, dimension=8, window=2, negatives=2, epochs=2, batch_size=4, seed=3
    )
    assert first.words == second.words
    np.testing.assert_array_equal(first.vectors, second.vectors)
    assert first.vectors.shape == (len(first.words), 8)
    assert np.isfinite(first.vectors).all()
    assert len(first.losses) == 2


def test_skipgram_rejects_bad_config_and_contextless_corpus():
    with pytest.raises(ValueError, match="must be positive"):
        SkipGramNS.train(["one two"], dimension=0)
    with pytest.raises(ValueError, match="context pairs"):
        SkipGramNS.train(["one", "two"], window=1)


def test_analogy_returns_exact_3cosadd_prediction_and_tracks_oov():
    words = ["alpha", "beta", "gamma", "delta", "outside"]
    expected = np.array([-1.0, 1.0, 1.0])
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            expected,
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = EmbeddingSpace(
        words=words,
        word_to_id={word: index for index, word in enumerate(words)},
        vectors=vectors,
        token_counts=np.ones(len(words), dtype=np.int64),
        elapsed_seconds=0.0,
        algorithm="test",
    )
    cases = [
        {"id": "in", "category": "toy", "a": "alpha", "b": "beta", "c": "gamma", "expected": "delta"},
        {"id": "oov", "category": "toy", "a": "missing", "b": "beta", "c": "gamma", "expected": "delta"},
    ]
    report = evaluate_analogies(model, cases)
    assert report["evaluated"] == 1
    assert report["coverage"] == 0.5
    assert report["cases"][0]["prediction"] == "delta"
    assert report["cases"][0]["correct"] is True
    assert report["cases"][1]["status"] == "oov"


def test_neighbor_metric_uses_only_available_queries():
    model = EmbeddingSpace(
        words=["apple", "fruit", "pear", "car"],
        word_to_id={"apple": 0, "fruit": 1, "pear": 2, "car": 3},
        vectors=np.array(
            [[1, 0], [0.99, 0.1], [0.95, 0.2], [-1, 0]], dtype=np.float32
        ),
        token_counts=np.ones(4, dtype=np.int64),
        elapsed_seconds=0,
        algorithm="test",
    )
    report = evaluate_neighbors(
        model,
        [
            {"id": "present", "category": "food", "word": "apple", "relevant": ["fruit", "pear"]},
            {"id": "missing", "category": "food", "word": "missing", "relevant": ["fruit"]},
        ],
        k=2,
    )
    assert report["evaluated"] == 1
    assert report["precision_at_k"] == 1.0
    assert nearest_neighbors(model, "absent") == []
