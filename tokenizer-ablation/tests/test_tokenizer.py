import unittest

from tokenizer_ablation.bpe import BytePairTokenizer, train_bpe
from tokenizer_ablation.experiment import evaluate, make_dataset, tokenizer_metrics


class TokenizerTests(unittest.TestCase):
    def test_byte_fallback_encodes_unicode_and_code(self) -> None:
        tokenizer = train_bpe(["customer_record customer_id"], vocab_size=300)
        encoded = tokenizer.encode("customer_record + Δ")
        self.assertTrue(encoded)
        self.assertGreaterEqual(tokenizer.vocab_size, 256)

    def test_vocab_size_is_capped_by_available_merges(self) -> None:
        tokenizer = train_bpe(["x"], vocab_size=1000)
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(tokenizer.encode("Δ"), list("Δ".encode("utf-8")))
        with self.assertRaisesRegex(ValueError, "between 256"):
            BytePairTokenizer(tokenizer.merges, 257)

    def test_training_is_deterministic(self) -> None:
        first = train_bpe(["fetch_customer_record", "save_invoice"], vocab_size=300)
        second = train_bpe(["fetch_customer_record", "save_invoice"], vocab_size=300)
        self.assertEqual(first.merges, second.merges)

    def test_budgeted_model_returns_accuracy_and_cost_metrics(self) -> None:
        train, test = make_dataset(seed=3)
        tokenizer = train_bpe([item.identifier for item in train[:100]], vocab_size=300)
        result = evaluate(
            tokenizer,
            train,
            test[:80],
            budget_kind="bytes",
            budget=1500,
            feature_width=64,
        )
        self.assertGreaterEqual(result["accuracy"], 0)
        self.assertLessEqual(result["accuracy"], 1)
        self.assertGreater(result["train_examples"], 0)
        self.assertGreater(result["training_seconds"], 0)
        metrics = tokenizer_metrics(tokenizer, test[:10])
        self.assertGreater(metrics["tokens_per_word"], 0)
        self.assertGreater(metrics["compression_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
