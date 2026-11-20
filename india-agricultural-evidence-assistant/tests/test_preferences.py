from __future__ import annotations

import json

import pytest

from app.preferences import load_ui_settings, save_ui_settings


def test_saved_settings_survive_reload_and_do_not_include_credentials(tmp_path):
    path = tmp_path / "data" / "ui-settings.json"
    save_ui_settings(
        path,
        {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "endpoint": "http://localhost:11434",
            "api_key": "must-not-be-saved",
        },
    )

    loaded = load_ui_settings(
        path,
        {"provider": "none", "model": "other", "endpoint": "http://localhost"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "endpoint": "http://localhost:11434",
    }
    assert "api_key" not in payload
    assert "must-not-be-saved" not in path.read_text(encoding="utf-8")


def test_missing_settings_use_environment_defaults(tmp_path):
    defaults = {"provider": "ollama", "model": "llama3.1:8b", "endpoint": "http://localhost:11434"}

    assert load_ui_settings(tmp_path / "missing.json", defaults) == defaults


@pytest.mark.parametrize(
    "values",
    [
        {"provider": "unknown", "model": "m", "endpoint": "http://localhost"},
        {"provider": "ollama", "model": "", "endpoint": "http://localhost"},
        {"provider": "ollama", "model": "m", "endpoint": ""},
    ],
)
def test_saving_invalid_settings_fails_explicitly(tmp_path, values):
    with pytest.raises(ValueError):
        save_ui_settings(tmp_path / "settings.json", values)


def test_corrupt_saved_settings_fail_explicitly(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not read saved app settings"):
        load_ui_settings(path, {"provider": "none", "model": "model", "endpoint": "http://localhost"})


def test_unsupported_saved_provider_is_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"provider": "unexpected", "model": "model", "endpoint": "http://localhost"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported provider"):
        load_ui_settings(path, {"provider": "none", "model": "model", "endpoint": "http://localhost"})
