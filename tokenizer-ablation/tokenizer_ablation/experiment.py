"""Code-identifier classification data and fixed-budget downstream experiments."""

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Example:
    identifier: str
    label: int


_ACTION_GROUPS = (
    ("read", ("fetch", "load", "read", "query", "get")),
    ("write", ("save", "store", "write", "update", "set")),
    ("delete", ("delete", "remove", "purge", "clear", "drop")),
    ("transform", ("parse", "format", "encode", "decode", "serialize")),
    ("auth", ("authenticate", "authorize", "verify", "validate", "sign")),
    ("compute", ("calculate", "compute", "derive", "evaluate", "aggregate")),
    ("network", ("send", "request", "connect", "dispatch", "publish")),
    ("test", ("mock", "assert", "fixture", "expect", "stub")),
)
LABEL_NAMES = tuple(name for name, _ in _ACTION_GROUPS)
_NOUNS = (
    "account user profile customer invoice payment transaction order product",
    "record document response request payload message event notification",
    "token credential session permission identity policy certificate",
    "value amount total balance rate score result metric summary",
    "cache index table row query schema field attribute property",
    "file stream buffer packet connection endpoint service client",
    "date timestamp interval timezone calendar schedule deadline",
    "address location region country language currency phone email",
    "model vector embedding feature label prediction cluster sample",
    "config option setting parameter context state status metadata",
    "database collection repository resource object entity",
)
_SUFFIXES = ("", "_by_id", "_from_cache", "_in_batch", "_for_user", "_response", "_metadata")


def make_dataset(seed: int = 17) -> tuple[list[Example], list[Example]]:
    nouns = tuple(dict.fromkeys(" ".join(_NOUNS).split()))
    examples: list[Example] = []
    rng = random.Random(seed)
    for label, (_, actions) in enumerate(_ACTION_GROUPS):
        identifiers: set[str] = set()
        while len(identifiers) < 1000:
            action, first, second, suffix = (
                rng.choice(actions),
                rng.choice(nouns),
                rng.choice(nouns),
                rng.choice(_SUFFIXES),
            )
            identifiers.add(f"{action}_{first}_{second}{suffix}")
        examples.extend(Example(identifier, label) for identifier in sorted(identifiers))
    random.Random(seed).shuffle(examples)
    split = int(len(examples) * 0.8)
    return examples[:split], examples[split:]


def make_tokenizer_corpus(
    excluded_identifiers: set[str], count: int = 35_000, seed: int = 1017
) -> list[str]:
    """Generate an unlabeled tokenizer corpus disjoint from classifier examples."""
    nouns = tuple(dict.fromkeys(" ".join(_NOUNS).split()))
    rng = random.Random(seed)
    identifiers: set[str] = set()
    actions = tuple(action for _, group in _ACTION_GROUPS for action in group)
    suffixes = _SUFFIXES + tuple(f"_v{index}" for index in range(100))
    while len(identifiers) < count:
        action, first, second, suffix = (
            rng.choice(actions),
            rng.choice(nouns),
            rng.choice(nouns),
            rng.choice(suffixes),
        )
        identifier = f"{action}_{first}_{second}{suffix}"
        if identifier not in excluded_identifiers:
            identifiers.add(identifier)
    return sorted(identifiers)


def _feature_vector(token_ids: list[int], width: int) -> np.ndarray:
    features = np.zeros(width, dtype=np.float32)
    for token_id in token_ids:
        raw = hashlib.blake2b(str(token_id).encode(), digest_size=8).digest()
        value = int.from_bytes(raw, "little")
        bucket = value % width
        features[bucket] += 1.0 if value & (1 << 63) else -1.0
    norm = np.linalg.norm(features)
    if norm:
        features /= norm
    return features


def _fit_classifier(
    tokenizer: Encoder,
    train_data: list[Example],
    *,
    budget_kind: str,
    budget: int,
    feature_width: int = 2048,
    epochs: int = 6,
    seed: int = 29,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    selected: list[tuple[np.ndarray, int, int, int]] = []
    used_bytes = 0
    used_tokens = 0
    for example in train_data:
        byte_count = len(example.identifier.encode("utf-8"))
        token_ids = tokenizer.encode(example.identifier)
        if selected and (
            (budget_kind == "bytes" and used_bytes + byte_count > budget)
            or (budget_kind == "tokens" and used_tokens + len(token_ids) > budget)
        ):
            break
        selected.append((_feature_vector(token_ids, feature_width), example.label, byte_count, len(token_ids)))
        used_bytes += byte_count
        used_tokens += len(token_ids)

    if not selected:
        raise ValueError("training budget is too small to include an example")
    class_count = max(item.label for item in train_data) + 1
    weights = np.zeros((class_count, feature_width), dtype=np.float32)
    bias = np.zeros(class_count, dtype=np.float32)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(len(selected))
        for index in order:
            features, label, _, _ = selected[int(index)]
            scores = weights @ features + bias
            scores -= np.max(scores)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            probabilities[label] -= 1.0
            learning_rate = 0.08 / (1.0 + len(selected) * (_ + 1) / 200_000)
            weights -= learning_rate * probabilities[:, None] * features[None, :]
            bias -= learning_rate * probabilities
    elapsed = time.perf_counter() - started
    return weights, bias, used_tokens, used_bytes, elapsed


def evaluate(
    tokenizer: Encoder,
    train_data: list[Example],
    test_data: list[Example],
    *,
    budget_kind: str,
    budget: int,
    feature_width: int = 2048,
) -> dict[str, Any]:
    weights, bias, used_tokens, used_bytes, train_seconds = _fit_classifier(
        tokenizer,
        train_data,
        budget_kind=budget_kind,
        budget=budget,
        feature_width=feature_width,
    )
    correct = 0
    for example in test_data:
        features = _feature_vector(tokenizer.encode(example.identifier), feature_width)
        correct += int(np.argmax(weights @ features + bias) == example.label)
    return {
        "accuracy": correct / len(test_data),
        "test_examples": len(test_data),
        "train_tokens": used_tokens,
        "train_bytes": used_bytes,
        "train_examples": min(len(train_data), _examples_under_budget(tokenizer, train_data, budget_kind, budget)),
        "training_seconds": train_seconds,
        "budget_kind": budget_kind,
        "budget": budget,
    }


def _examples_under_budget(
    tokenizer: Encoder, data: list[Example], budget_kind: str, budget: int
) -> int:
    total = 0
    count = 0
    for example in data:
        size = (
            len(tokenizer.encode(example.identifier))
            if budget_kind == "tokens"
            else len(example.identifier.encode("utf-8"))
        )
        if count and total + size > budget:
            break
        total += size
        count += 1
    return count


def tokenizer_metrics(tokenizer: Encoder, examples: list[Example]) -> dict[str, float]:
    token_count = 0
    byte_count = 0
    word_count = 0
    for example in examples:
        token_count += len(tokenizer.encode(example.identifier))
        byte_count += len(example.identifier.encode("utf-8"))
        word_count += len(re.findall(r"[A-Za-z0-9]+", example.identifier))
    return {
        "tokens_per_word": token_count / word_count if word_count else 0.0,
        "bytes_per_token": byte_count / token_count if token_count else 0.0,
        "compression_ratio": byte_count / token_count if token_count else 0.0,
    }
