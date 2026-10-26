"""Training, evaluation, and synthetic long-range recall experiments."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sequence_models.models import MODEL_NAMES, ModelConfig, build_model, count_parameters


def encode_corpus(path: Path, repeat: int = 20) -> tuple[torch.Tensor, torch.Tensor, str]:
    if repeat < 1:
        raise ValueError("corpus repeat must be positive")
    text = path.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("corpus must contain at least 100 characters")
    alphabet = "".join(sorted(set(text)))
    char_to_id = {character: index for index, character in enumerate(alphabet)}
    encoded = torch.tensor([char_to_id[character] for character in text], dtype=torch.long)
    split = int(len(encoded) * 0.9)
    train_base, valid_base = encoded[:split], encoded[split:]
    return train_base.repeat(repeat), valid_base.repeat(repeat), alphabet


def _language_batch(
    data: torch.Tensor, batch_size: int, sequence_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))
    inputs = torch.stack([data[start : start + sequence_length] for start in starts])
    targets = torch.stack([data[start + 1 : start + sequence_length + 1] for start in starts])
    return inputs.to(device), targets.to(device)


def _evaluate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets = _language_batch(data, batch_size, sequence_length, device)
            logits = model(inputs)
            losses.append(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def sample_text(
    model: nn.Module,
    seed: torch.Tensor,
    alphabet: str,
    length: int = 200,
    temperature: float = 0.8,
) -> str:
    if temperature <= 0 or not alphabet or seed.ndim != 1:
        raise ValueError("temperature and alphabet must be positive/non-empty; seed must be 1D")
    device = next(model.parameters()).device
    tokens = seed.to(device).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        for _ in range(length):
            context = tokens[:, -model.max_sequence_length :]
            logits = model(context)[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
    return "".join(alphabet[index] for index in tokens[0].tolist())


def measure_generation_speed(
    model: nn.Module,
    seed: torch.Tensor,
    token_count: int = 100,
) -> dict[str, float]:
    device = next(model.parameters()).device
    tokens = seed.to(device).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(tokens[:, -model.max_sequence_length :])
        started = time.perf_counter()
        for _ in range(token_count):
            logits = model(tokens[:, -model.max_sequence_length :])
            tokens = torch.cat((tokens, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        elapsed = time.perf_counter() - started
    return {
        "tokens_per_second": token_count / elapsed if elapsed else 0.0,
        "mean_latency_ms": elapsed * 1000 / token_count,
        "tokens_measured": token_count,
    }


def train_language_model(
    config: ModelConfig,
    train_data: torch.Tensor,
    valid_data: torch.Tensor,
    alphabet: str,
    *,
    target_parameters: int,
    epochs: int = 3,
    steps_per_epoch: int = 25,
    batch_size: int = 16,
    sequence_length: int = 64,
    device: torch.device | None = None,
    seed: int = 17,
) -> tuple[nn.Module, dict[str, Any], str]:
    if min(epochs, steps_per_epoch, batch_size, sequence_length) < 1:
        raise ValueError("training parameters must be positive")
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = build_model(config.name, len(alphabet), config, sequence_length + 1).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history: list[dict[str, float]] = []
    epoch_times = []
    step = 0
    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        for _ in range(steps_per_epoch):
            inputs, targets = _language_batch(train_data, batch_size, sequence_length, device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits.reshape(-1, len(alphabet)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            history.append({"step": float(step), "train_loss": float(loss.item())})
        epoch_seconds = time.perf_counter() - started
        epoch_times.append(epoch_seconds)
        validation_loss = _evaluate_loss(
            model, valid_data, batch_size, sequence_length, batches=8, device=device
        )
        history.append({"step": float(step), "validation_loss": validation_loss})
    prompt = train_data[: min(16, len(train_data))]
    sample = sample_text(model, prompt, alphabet, length=200)
    speed = measure_generation_speed(model, prompt)
    result = {
        "model": config.name,
        "parameters": parameter_count,
        "target_parameters": target_parameters,
        "parameter_delta_percent": 100 * (parameter_count - target_parameters) / target_parameters,
        "config": {
            "hidden_size": config.hidden_size,
            "layers": config.layers,
            "heads": config.heads if config.name == "transformer" else None,
            "feedforward_size": config.feedforward_size if config.name == "transformer" else None,
        },
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "training_seconds_per_epoch": epoch_times,
        "final_validation_loss": _evaluate_loss(
            model, valid_data, batch_size, sequence_length, batches=16, device=device
        ),
        "history": history,
        "generation": speed,
        "perplexity": math.exp(min(20, history[-1]["validation_loss"])),
    }
    return model, result, sample


def make_recall_data(
    examples: int, sequence_length: int = 101, seed: int = 31
) -> tuple[torch.Tensor, torch.Tensor]:
    if examples < 1 or sequence_length < 2:
        raise ValueError("recall example count must be positive and sequence length at least two")
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randint(3, 16, (examples, sequence_length), generator=generator)
    labels = torch.randint(0, 2, (examples,), generator=generator)
    inputs[:, 0] = labels
    inputs[:, -1] = 2
    return inputs, labels


def train_recall_model(
    config: ModelConfig,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    valid_inputs: torch.Tensor,
    valid_labels: torch.Tensor,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    device: torch.device | None = None,
    seed: int = 41,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = build_model(config.name, 16, config, train_inputs.shape[1], task="recall").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    epoch_times = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_inputs))
        started = time.perf_counter()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            inputs = train_inputs[indices].to(device)
            labels = train_labels[indices].to(device)
            loss = nn.functional.cross_entropy(model(inputs), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epoch_times.append(time.perf_counter() - started)
    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(valid_inputs), batch_size):
            inputs = valid_inputs[start : start + batch_size].to(device)
            labels = valid_labels[start : start + batch_size].to(device)
            correct += int((model(inputs).argmax(dim=-1) == labels).sum().item())
    return {
        "model": config.name,
        "accuracy": correct / len(valid_inputs),
        "random_baseline": 0.5,
        "sequence_length": int(train_inputs.shape[1]),
        "remembered_position": 0,
        "epochs": epochs,
        "training_seconds_per_epoch": epoch_times,
        "parameters": count_parameters(model),
    }
