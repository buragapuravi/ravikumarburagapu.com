import json
from pathlib import Path

import pytest

from word2vec_lab.data import load_analogies, load_corpus, load_neighbors
from word2vec_lab.model import SkipGramNS


def test_sample_fixtures_have_fifty_analogies_and_twenty_neighbors():
    sample_dir = Path(__file__).parents[1] / "samples"
    assert len(load_analogies(sample_dir / "analogies.jsonl")) == 50
    assert len(load_neighbors(sample_dir / "neighbor-pairs.jsonl")) == 20
    assert len(load_corpus(sample_dir / "demo-corpus.txt")) >= 5


def test_demo_corpus_covers_all_analogy_words_and_neighbor_queries():
    sample_dir = Path(__file__).parents[1] / "samples"
    model = SkipGramNS.train(
        load_corpus(sample_dir / "demo-corpus.txt"),
        dimension=8,
        window=2,
        negatives=2,
        epochs=1,
        batch_size=128,
        seed=11,
    )
    analogies = load_analogies(sample_dir / "analogies.jsonl")
    neighbors = load_neighbors(sample_dir / "neighbor-pairs.jsonl")
    analogy_terms = {
        row[key]
        for row in analogies
        for key in ("a", "b", "c", "expected")
    }
    neighbor_terms = {row["word"] for row in neighbors}
    assert analogy_terms <= model.word_to_id.keys()
    assert neighbor_terms <= model.word_to_id.keys()


def test_loaders_reject_duplicate_ids_and_invalid_neighbor_labels(tmp_path):
    analogies = tmp_path / "analogies.jsonl"
    row = {
        "id": "one",
        "category": "x",
        "a": "a",
        "b": "b",
        "c": "c",
        "expected": "d",
    }
    analogies.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate id"):
        load_analogies(analogies)

    neighbors = tmp_path / "neighbors.jsonl"
    neighbors.write_text(
        json.dumps(
            {"id": "one", "category": "x", "word": "anchor", "relevant": []}
        )
    )
    with pytest.raises(ValueError, match="invalid category"):
        load_neighbors(neighbors)


def test_corpus_reader_rejects_empty_file(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("\n  \n")
    with pytest.raises(ValueError, match="no non-empty"):
        load_corpus(corpus)
