from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = ROOT / ".codex" / "agents"

EXPECTED_MODELS = {
    "codebase-scout.toml": "gpt-5.6-luna",
    "codebase-explorer.toml": "gpt-5.6-terra",
    "issue-implementer.toml": "gpt-5.6-sol",
    "test-writer.toml": "gpt-5.6-sol",
    "issue-validator.toml": "gpt-5.6-sol",
}


def test_custom_agents_use_canonical_model_slugs() -> None:
    actual: dict[str, str] = {}

    for filename, expected_model in EXPECTED_MODELS.items():
        with (AGENTS_DIR / filename).open("rb") as handle:
            config = tomllib.load(handle)

        model = config.get("model")
        assert isinstance(model, str)
        assert model == expected_model
        actual[filename] = model

    assert "gpt-5.6" not in actual.values()
