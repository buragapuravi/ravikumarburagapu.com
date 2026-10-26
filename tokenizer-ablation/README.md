# Tokenizer Ablation

How much does tokenizer choice change downstream accuracy when the classifier is held constant? This reproducible experiment trains a byte-level BPE tokenizer from scratch, sweeps vocabulary sizes, and evaluates code-identifier classification at fixed token and UTF-8 byte budgets.

> **Scope:** This is a controlled toy experiment, not a general claim about LLM quality. It uses generated code identifiers and a small hashed bag-of-subword classifier. Results measure this task and setup only.

## Quick start

Requires Python 3.11+.

```sh
cd tokenizer-ablation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
tokenizer-ablation --output results/ablation.json
pytest
```

The default sweep trains BPE tokenizers with vocabularies of 1k, 4k, 16k, and 32k tokens on a separate, reproducible 35,000-identifier unlabeled corpus, then trains the same fixed-width classifier for each tokenizer under both 10k-token and 50k-byte budgets. The tokenizer corpus excludes all labeled classifier examples. The report includes downstream accuracy, examples consumed, actual tokens/bytes consumed, and classifier training time. It also reports tokens per identifier word and UTF-8 bytes per token on held-out identifiers.

Vocabulary size includes all 256 byte-fallback tokens. If the training corpus cannot support a requested vocabulary, the report gives the actual learned size rather than inserting unused filler tokens.

## Optional comparison tokenizers

```sh
pip install -e ".[tiktoken,transformers]"
tokenizer-ablation --with-references --output results/with-references.json
```

This adds `tiktoken`'s `cl100k_base` and the uncased BERT WordPiece tokenizer from Hugging Face Transformers (`bert-base-uncased`). The reference tokenizer weights may need to be downloaded. Missing reference dependencies or model files are reported in `unavailable_references`; no fallback tokenizer is substituted.

## Experiment design

- **Corpus:** deterministic, generated code-like identifiers across eight operation classes (read, write, delete, transform, auth, compute, network, test). BPE trains on a separate unlabeled identifier corpus; the classifier uses a seeded, disjoint labeled train/test split.
- **From-scratch tokenizer:** byte fallback plus learned adjacent-pair merges, with deterministic tie-breaking. Implementation is in `tokenizer_ablation/bpe.py`.
- **Downstream task:** classify an identifier into one of eight operation classes.
- **Tiny model:** multinomial logistic classifier over a fixed-width, signed-hashed bag of subword IDs. Architecture, feature width, optimizer, epochs, and seed are held constant across tokenizers.
- **Fixed token budget:** examples are consumed in a fixed order until the next example would exceed the token limit. Thus tokenizers can see different numbers of examples; both counts are reported. Budgets are upper bounds if the training split is exhausted first.
- **Fixed byte budget:** the same byte-accounting rule is applied to UTF-8 source identifiers. Compare both modes to see the effect of token efficiency versus text volume.
- **Timing:** classifier-fit time only; it excludes tokenizer training, feature extraction, and evaluation. BPE tokenizer training time is reported separately.

The output is JSON to stdout by default, or to a path passed with `--output`. Adjust `--vocab-sizes`, `--token-budget`, `--byte-budget`, `--feature-width`, `--tokenizer-corpus-limit`, and `--seed` to reproduce variations. Smaller tokenizer corpus limits can prevent the largest requested vocabularies from being reached; actual vocabulary sizes are always reported.

## Example result

One run with the default seed and settings produced:

| BPE vocabulary | Tokens / identifier word | UTF-8 bytes / token | Accuracy at 10k-token cap | Accuracy at 50k-byte cap |
|---:|---:|---:|---:|---:|
| 1,000 | 0.936 | 7.013 | 0.959 | 0.949 |
| 4,000 | 0.750 | 8.747 | 0.467 | 0.452 |
| 16,000 | 0.524 | 12.527 | 0.339 | 0.243 |
| 32,000 | 0.500 | 13.132 | 0.349 | 0.239 |

Here the larger vocabularies compress the identifiers into fewer tokens but generalize worse on this held-out synthetic classification task. This illustrates a trade-off in this particular hashed classifier and generated corpus; it is not evidence that smaller tokenizers are generally better. Repeat with different seeds and real task data before drawing conclusions.

## Interpretation and limitations

Token/word and byte/token ratios depend on the selected code-like corpus. A vocabulary-size change also changes the token IDs entering a fixed-width hashing layer, so the experiment measures the full effect on this specific classifier, not only sequence length. The generated task uses lexical cues by design; it does not test compositional arithmetic, character counting, multilingual generalization, or pretrained LLM behavior. Run multiple seeds and use a real, appropriately licensed corpus before drawing broader conclusions.

## License

MIT. See [LICENSE](LICENSE).
