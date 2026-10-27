# Word2Vec Lab: From-Scratch Skip-Gram + Analogy Geometry

Train **skip-gram with negative sampling (SGNS)** in NumPy, sweep embedding dimension, context window, and negative sample count, then inspect analogy and hand-labeled nearest-neighbor results. The default experiment compares 27 configurations: dimensions `50, 100, 300` × windows `2, 5, 10` × negatives `5, 10, 15`.

> **Corpus and evaluation notice:** The bundled corpus, 50 analogy questions, and 20 neighborhood relevance sets are original illustrative fixtures. They are not the author's private papers or talks, a standard analogy benchmark, or external human validation. The small demo corpus is intentionally not evidence that any famous analogy works. Use a corpus you own or are authorized to process and review/replace labels for your domain before reporting results.

## Quick start

Requires Python 3.11+ and NumPy.

```sh
cd word2vec-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
word2vec-lab
```

Default results are written to `results/` (ignored by Git):

- `report.json`: full per-configuration analogy and neighbor predictions, category metrics, training losses, runtime, configuration, and OOV coverage.
- `summary.csv`: compact sweep results suitable for plotting.

To try a smaller run while developing:

```sh
word2vec-lab --dimensions 50 --windows 2 --negative-samples 5 --epochs 2
```

## Algorithm

The implementation builds center/context pairs from sentence-bounded sliding windows, initializes separate input/output embeddings, samples negatives from the unigram distribution raised to `0.75`, and minimizes the logistic negative-sampling objective with mini-batch NumPy updates. The optimizer uses a linearly decayed learning rate. It does not silently join sentences across lines.

Analogy prediction uses normalized `b - a + c` (3CosAdd), excluding the three query words from candidates. Reports include both accuracy on in-vocabulary cases and coverage across the full fixture; OOV cases are not misreported as incorrect model predictions or silently dropped without a count.

Nearest-neighbor `precision@k` compares cosine top-k results with the hand-specified relevant-word set. The bundled labels are demo judgments; adapt them with domain reviewers. Training time measures corpus tokenization, pair/co-occurrence construction, and model fitting; it does not include fixture loading or evaluation.

## Optional baselines and visualization

```sh
pip install -e ".[fasttext,viz]"
word2vec-lab --baselines --visualize
```

- **GloVe:** sparse co-occurrence GloVe-style objective implemented in this project, trained on the same corpus at each selected dimension and a fixed window of 5.
- **FastText:** Gensim FastText skip-gram, trained on the same corpus at each dimension and fixed window 5. FastText uses subword character n-grams; install the `fasttext` extra to enable it.
- **t-SNE:** optional SVG visualization of the selected SGNS run's in-vocabulary analogy words, colored by the fixture's semantic category. Install the `viz` extra. t-SNE is a stochastic 2D projection for exploration; distances and clusters in the plot are not a validation metric.

Baselines use the same text and minimum count, but their objectives differ; GloVe training epochs, FastText epochs, and SGNS pair updates are not computationally equivalent. Treat runtime as an observed implementation measurement, not a controlled hardware-independent cost comparison.

## Use an authorized corpus

Provide a UTF-8 text file with one sentence or document line per record:

```sh
word2vec-lab \
  --corpus /path/to/authorized-corpus.txt \
  --analogies /path/to/your-analogies.jsonl \
  --neighbors /path/to/your-neighbor-pairs.jsonl \
  --output-dir results/domain-run
```

Analogy JSONL records have `id`, `category`, `a`, `b`, `c`, and `expected` fields. Neighbor records have `id`, `category`, `word`, and a non-empty `relevant` word list. Keep data local unless you intentionally publish it; output predictions can disclose corpus vocabulary.

## Options

```sh
word2vec-lab --help
word2vec-lab --dimensions 50,100,300 \
  --windows 2,5,10 --negative-samples 5,10,15 \
  --epochs 5 --min-count 1 --batch-size 256 --seed 17
```

## License

MIT. See [LICENSE](LICENSE).
