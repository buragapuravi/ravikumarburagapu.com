"""Parameter-matched recurrent and causal Transformer architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

MODEL_NAMES = ("rnn", "lstm", "gru", "transformer")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_size: int
    layers: int
    heads: int = 4
    feedforward_size: int | None = None


class SequenceLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig, max_sequence_length: int) -> None:
        super().__init__()
        self.config = config
        self.name = config.name
        self.max_sequence_length = max_sequence_length
        hidden = config.hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden)
        if config.name in {"rnn", "lstm", "gru"}:
            recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[config.name]
            options = {"nonlinearity": "tanh"} if config.name == "rnn" else {}
            self.backbone = recurrent(hidden, hidden, num_layers=config.layers, batch_first=True, **options)
            self.position_embedding = None
        elif config.name == "transformer":
            self.position_embedding = nn.Embedding(max_sequence_length, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.heads,
                dim_feedforward=config.feedforward_size or hidden * 4,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        else:
            raise ValueError(f"unknown model: {config.name}")
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        embedded = self.embedding(tokens)
        if self.position_embedding is not None:
            if length > self.max_sequence_length:
                raise ValueError("input sequence exceeds configured maximum length")
            positions = torch.arange(length, device=tokens.device)
            embedded = embedded + self.position_embedding(positions)[None, :, :]
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
            )
            hidden = self.backbone(embedded, mask=mask)
        else:
            hidden, _ = self.backbone(embedded)
        return self.output(hidden)


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size: int, class_count: int, config: ModelConfig, max_sequence_length: int) -> None:
        super().__init__()
        self.config = config
        self.max_sequence_length = max_sequence_length
        hidden = config.hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden)
        if config.name in {"rnn", "lstm", "gru"}:
            recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[config.name]
            self.backbone = recurrent(hidden, hidden, num_layers=config.layers, batch_first=True)
            self.position_embedding = None
        else:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.heads,
                dim_feedforward=config.feedforward_size or hidden * 4,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, class_count)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        if length > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured maximum length")
        hidden = self.embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(length, device=tokens.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
            hidden = self.backbone(hidden, mask=mask)
        else:
            hidden, _ = self.backbone(hidden)
        return self.output(hidden[:, -1])


def _count_lm_parameters(
    name: str, vocab_size: int, config: ModelConfig, max_sequence_length: int
) -> int:
    hidden = config.hidden_size
    shared = vocab_size * hidden + hidden * vocab_size + vocab_size
    if name == "rnn":
        recurrent = config.layers * (2 * hidden * hidden + 2 * hidden)
    elif name == "gru":
        recurrent = config.layers * (6 * hidden * hidden + 6 * hidden)
    elif name == "lstm":
        recurrent = config.layers * (8 * hidden * hidden + 8 * hidden)
    else:
        recurrent = hidden * max_sequence_length + config.layers * (12 * hidden * hidden + 13 * hidden)
    return shared + recurrent


def matched_configs(
    vocab_size: int,
    target_parameters: int = 1_000_000,
    recurrent_layers: int = 1,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    max_sequence_length: int = 65,
) -> dict[str, ModelConfig]:
    if target_parameters < 10_000 or max_sequence_length < 1:
        raise ValueError("target parameter count must be at least 10000")
    configs: dict[str, ModelConfig] = {}
    for name in MODEL_NAMES:
        if name == "transformer":
            upper = int((target_parameters / (12 * transformer_layers)) ** 0.5 * 2)
            candidates = range(transformer_heads, max(transformer_heads + 1, upper), transformer_heads)
        else:
            candidates = range(8, max(9, int(target_parameters**0.5 * 2)))
        best: tuple[float, ModelConfig] | None = None
        for hidden in candidates:
            config = ModelConfig(
                name=name,
                hidden_size=hidden,
                layers=transformer_layers if name == "transformer" else recurrent_layers,
                heads=transformer_heads,
                feedforward_size=hidden * 4,
            )
            count = _count_lm_parameters(name, vocab_size, config, max_sequence_length)
            score = abs(count - target_parameters)
            if best is None or score < best[0]:
                best = (score, config)
        assert best is not None
        configs[name] = best[1]
    return configs


def build_model(
    name: str,
    vocab_size: int,
    config: ModelConfig,
    max_sequence_length: int = 128,
    task: str = "language",
) -> nn.Module:
    if name != config.name:
        raise ValueError("model name must match ModelConfig.name")
    if task == "language":
        return SequenceLanguageModel(vocab_size, config, max_sequence_length)
    if task == "recall":
        return SequenceClassifier(vocab_size, 2, config, max_sequence_length)
    raise ValueError("task must be 'language' or 'recall'")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
