"""Optional t-SNE visualization for analogy vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from word2vec_lab.evaluation import EmbeddingModel


def plot_analogy_tsne(
    model: EmbeddingModel, cases: list[dict[str, Any]], output: Path, *, seed: int = 17
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError(
            "t-SNE visualization requires optional dependencies; install with "
            "`pip install -e '.[viz]'`"
        ) from error

    categories: dict[str, str] = {}
    for case in cases:
        for field in ("a", "b", "c", "expected"):
            word = case[field].casefold()
            if word in model.word_to_id:
                categories.setdefault(word, case["category"])
    words = sorted(categories)
    if len(words) < 3:
        raise ValueError("t-SNE needs at least three in-vocabulary analogy words")
    matrix = np.asarray([model.vectors[model.word_to_id[word]] for word in words])
    perplexity = min(30.0, max(2.0, float(len(words) - 1) / 3.0))
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(matrix)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13, 9))
    for category in sorted(set(categories.values())):
        selected = [index for index, word in enumerate(words) if categories[word] == category]
        axis.scatter(projection[selected, 0], projection[selected, 1], label=category, s=28)
    for index, word in enumerate(words):
        axis.annotate(word, projection[index], fontsize=7, alpha=0.8)
    axis.set_title(f"Analogy vocabulary t-SNE ({model.vectors.shape[1]} dimensions)")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
