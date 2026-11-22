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
            f'\nCODEX_MODEL="{WORKER_DEFAULT_MODEL}"', '\nCODEX_MODEL="configured-sol"'
        )
        .replace(
            f'CODEX_ALTERNATE_MODEL="{WORKER_ALTERNATE_MODEL}"',
            'CODEX_ALTERNATE_MODEL="configured-astra"',
        )
    )
    monkeypatch.setattr(settings, "DEFAULT_CONFIG", config)
    repo_choices = settings.implementation_models(ops / "lib/agents.conf")
    assert settings.implementation_models() == ("configured-sol", *repo_choices[1:])
    assert settings.implementation_model(None) == "configured-sol"
    assert settings.implementation_model(None, previous=OLD_SOL) == OLD_SOL
    for model in (OLD_ASTRA, WORKER_ALTERNATE_MODEL):
        with pytest.raises(ValueError, match="configured Sol"):
            settings.implementation_model(None, previous=model)
    with pytest.raises(ValueError, match="continuation-only"):
        settings.implementation_model(OLD_SOL)

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
        assert (
            root.parse_args([*args, "--model", "configured-sol"]).model
            == "configured-sol"
        )
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
        dispatcher.parse_args([*args, "--model", "configured-sol"]).model
        == "configured-sol"
    )
    # The actual runner resolver sees the same new pins; profile pins stay separate.
    assert settings.resolve(config, "codex", None, "configured-sol", None, "write")[
        "selected"
    ] == {
        "model": "configured-sol",
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


def test_selectable_models_are_explicit_new_worker_choices_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selectable model starts and continues a worker but never becomes the default."""
    ops = Path(__file__).resolve().parents[3] / ".agents/ops"
    monkeypatch.syspath_prepend(str(ops))
    settings = importlib.import_module("lib.agent_settings")
    base = (ops / "lib/agents.conf").read_text()
    config = tmp_path / "agents.conf"

    def write(selectable: str) -> None:
        lines = [
            line
            for line in base.splitlines()
            if not line.startswith(settings.SELECTABLE_MODELS_PIN + "=")
        ]
        lines.append(f'{settings.SELECTABLE_MODELS_PIN}="{selectable}"')
        config.write_text("\n".join(lines) + "\n")

    write("chosen-b,chosen-a")
    monkeypatch.setattr(settings, "DEFAULT_CONFIG", config)
    assert settings.implementation_models() == (
        WORKER_DEFAULT_MODEL,
        "chosen-a",
        "chosen-b",
    )
    assert settings.implementation_model(None) == WORKER_DEFAULT_MODEL
    assert settings.implementation_model("chosen-a") == "chosen-a"
    assert settings.implementation_model(None, previous="chosen-b") == "chosen-b"
    with pytest.raises(ValueError, match="cannot change"):
        settings.implementation_model("chosen-a", previous=WORKER_DEFAULT_MODEL)
    with pytest.raises(ValueError, match="configured Sol"):
        settings.implementation_model("unlisted-model")
    assert settings.resolve(config, "codex", None, "chosen-a", None, "write")[
        "selected"
    ] == {"model": "chosen-a", "effort": "max", "service_tier": "default"}
    with pytest.raises(ValueError, match="effort must be max"):
        settings.resolve(config, "codex", None, "chosen-a", "high", "write")

    root = importlib.import_module("parent.cli_impl").parser()
    start = [
        "--parent",
        "1",
        "start-worker",
        "--child",
        "10",
        "--prompt-file",
        "task.md",
        "--branch",
        "codex/worker",
        "--worktree",
        "/fixture",
    ]
    assert root.parse_args([*start, "--model", "chosen-b"]).model == "chosen-b"

    for invalid, message in (
        (WORKER_DEFAULT_MODEL, "must not repeat CODEX_MODEL"),
        (OLD_SOL, "overlap"),
        ("chosen-a,chosen-a", "invalid model list"),
    ):
        write(invalid)
        with pytest.raises(ValueError, match=message):
            if message == "overlap":
                settings.implementation_model(None)
            else:
                settings.implementation_models()
