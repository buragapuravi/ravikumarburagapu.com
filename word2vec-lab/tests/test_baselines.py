import numpy as np

from word2vec_lab.baselines import train_glove


def test_glove_baseline_builds_finite_vectors_for_same_corpus():
    model = train_glove(
        ["red apple grows", "green apple grows", "red berry falls"],
        dimension=6,
        window=2,
        epochs=2,
        seed=7,
    )
    assert model.algorithm == "glove"
    assert model.vectors.shape == (len(model.words), 6)
    assert np.isfinite(model.vectors).all()
