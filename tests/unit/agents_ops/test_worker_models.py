"""Implementation model choices share the runner config and continuation policy."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.support.agents_ops.model_pins import (
    OLD_ASTRA,
    OLD_SOL,
    REVIEW_FIXED_MODEL,
    WORKER_ALTERNATE_MODEL,
    WORKER_DEFAULT_MODEL,
)


def test_implementation_choices_follow_runner_config_without_parser_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = Path(__file__).resolve().parents[3] / ".agents/ops"
    monkeypatch.syspath_prepend(str(ops))
    settings = importlib.import_module("lib.agent_settings")
    config = tmp_path / "agents.conf"
    config.write_text(
        (ops / "lib/agents.conf")
        .read_text()
        .replace(
            f'CODEX_MODEL="{WORKER_DEFAULT_MODEL}"', 'CODEX_MODEL="configured-sol"'
        )
        .replace(
            f'CODEX_ALTERNATE_MODEL="{WORKER_ALTERNATE_MODEL}"',
            'CODEX_ALTERNATE_MODEL="configured-astra"',
        )
    )
    monkeypatch.setattr(settings, "DEFAULT_CONFIG", config)
    assert settings.implementation_models() == ("configured-sol", "configured-astra")
    assert settings.implementation_model(None) == "configured-sol"
    assert (
        settings.implementation_model(None, previous="configured-astra")
        == "configured-astra"
    )
    assert (
        settings.implementation_model("configured-astra", previous="configured-astra")
        == "configured-astra"
    )
    with pytest.raises(ValueError, match="cannot change"):
        settings.implementation_model("configured-sol", previous="configured-astra")
    for previous in (OLD_SOL, OLD_ASTRA):
        assert settings.implementation_model(None, previous=previous) == previous
        with pytest.raises(ValueError, match="continuation-only"):
            settings.implementation_model(previous)
    with pytest.raises(ValueError, match="default or configured alternate"):
        settings.implementation_model(None, previous="unlisted-model")

    cli = importlib.import_module("parent.cli_impl")
    root = cli.parser()
    for command, required in (
        ("run-worker", []),
        ("start-worker", ["--branch", "codex/worker", "--worktree", "/fixture"]),
        ("continue-worker", ["--evidence-file", "evidence.md"]),
    ):
        args = [
            "--parent",
            "1",
            command,
            "--child",
            "10",
            "--prompt-file",
            "task.md",
            *required,
        ]
        assert (
            root.parse_args(args).model is None
        )  # Resume must never acquire a parser default.
        for model in ("configured-sol", "configured-astra"):
            assert root.parse_args([*args, "--model", model]).model == model
        if command == "start-worker":
            with pytest.raises(SystemExit) as rejected:
                root.parse_args([*args, "--model", OLD_ASTRA])
            assert rejected.value.code == 2
        else:
            assert root.parse_args([*args, "--model", OLD_ASTRA]).model == OLD_ASTRA

    dispatcher = importlib.import_module("dispatcher.cli_impl").parser()
    args = [
        "start-child",
        "--parent",
        "1",
        "--child",
        "10",
        "--delegation-id",
        "fixture",
        "--prompt-file",
        "task.md",
    ]
    assert (
        dispatcher.parse_args([*args, "--model", "configured-astra"]).model
        == "configured-astra"
    )
    # The actual runner resolver sees the same new pins; profile pins stay separate.
    assert settings.resolve(config, "codex", None, "configured-astra", None, "write")[
        "selected"
    ] == {
        "model": "configured-astra",
        "effort": "max",
        "service_tier": "default",
    }
    review = importlib.import_module("lib.review_settings")
    choice = review.choose(
        review.load_policy(config),
        model=REVIEW_FIXED_MODEL,
        reason="実装workerのモデル設定とは独立した正式レビューの選択を確認する。",
    )
    assert settings.resolve(
        config, "codex", "review", None, None, "read-only", review_selection=choice
    )["selected"] == {
        "model": REVIEW_FIXED_MODEL,
        "effort": "xhigh",
        "service_tier": "default",
    }
