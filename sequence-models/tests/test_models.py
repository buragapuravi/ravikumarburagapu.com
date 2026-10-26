import unittest

import torch

from sequence_models.experiment import make_recall_data
from sequence_models.models import (
    MODEL_NAMES,
    build_model,
    count_parameters,
    matched_configs,
)


class SequenceModelTests(unittest.TestCase):
    def test_parameter_matching_is_close_to_target(self) -> None:
        configs = matched_configs(
            vocab_size=64, target_parameters=1_000_000, max_sequence_length=65
        )
        counts = {}
        for name in MODEL_NAMES:
            model = build_model(name, 64, configs[name], max_sequence_length=65)
            counts[name] = count_parameters(model)
            self.assertLess(abs(counts[name] - 1_000_000) / 1_000_000, 0.02)

    def test_language_models_return_per_position_logits(self) -> None:
        configs = matched_configs(vocab_size=32, target_parameters=40_000)
        tokens = torch.randint(0, 32, (2, 12))
        for name in MODEL_NAMES:
            model = build_model(name, 32, configs[name], max_sequence_length=16)
            self.assertEqual(tuple(model(tokens).shape), (2, 12, 32))

    def test_recall_data_places_answer_at_first_position(self) -> None:
        tokens, labels = make_recall_data(20, sequence_length=101, seed=4)
        self.assertEqual(tuple(tokens.shape), (20, 101))
        self.assertTrue(torch.equal(tokens[:, 0], labels))
        self.assertTrue(torch.all(tokens[:, -1] == 2))
        self.assertTrue(torch.all((tokens[:, 1:-1] >= 3) & (tokens[:, 1:-1] < 16)))

    def test_transformer_uses_causal_context(self) -> None:
        config = matched_configs(vocab_size=16, target_parameters=30_000)["transformer"]
        model = build_model("transformer", 16, config, max_sequence_length=8).eval()
        first = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 7, 8]])
        with torch.inference_mode():
            original_logits = model(first)
            changed_logits = model(changed_future)
        torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])


if __name__ == "__main__":
    unittest.main()
