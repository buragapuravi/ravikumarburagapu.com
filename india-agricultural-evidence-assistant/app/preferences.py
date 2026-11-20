from __future__ import annotations

import json
import os
from pathlib import Path

_PROVIDERS = {"none", "ollama", "openai-compatible"}


def load_ui_settings(path: Path, defaults: dict[str, str]) -> dict[str, str]:
    if not path.exists():
        return dict(defaults)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read saved app settings at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Saved app settings must contain a JSON object: {path}")
    provider = payload.get("provider", defaults["provider"])
    model = payload.get("model", defaults["model"])
    endpoint = payload.get("endpoint", defaults["endpoint"])
    if provider not in _PROVIDERS:
        raise ValueError(f"Saved app settings use an unsupported provider: {provider!r}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Saved app model name must be a non-empty string.")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Saved app endpoint must be a non-empty string.")
    return {
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
    }


def save_ui_settings(path: Path, values: dict[str, str]) -> None:
    provider = values.get("provider", "")
    model = values.get("model", "")
    endpoint = values.get("endpoint", "")
    if provider not in _PROVIDERS:
        raise ValueError(f"Unsupported answer provider: {provider!r}")
    if not model.strip():
        raise ValueError("Enter a model name before saving app settings.")
    if not endpoint.strip():
        raise ValueError("Enter an endpoint URL before saving app settings.")
    payload = {
        "provider": provider,
        "model": model.strip(),
        "endpoint": endpoint.strip(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    except OSError as error:
        raise RuntimeError(f"Could not save app settings at {path}: {error}") from error
