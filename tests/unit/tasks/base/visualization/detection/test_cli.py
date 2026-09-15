"""Focused coverage for the shared detection CLI and its four task scripts.

Each review/inference script owns its ``argparse`` parser and validates its
module-level ``NonHydraPathBoundary`` before the shared launcher imports a task
service, so these tests pin both the runtime ordering and the source-level
boundary binding the configuration audit discovers.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from src.utils.configuration import PathContractError
from src.utils.configuration.discovery import discover_runtime_boundaries
from src.utils.configuration.inventory import DEFAULT_AUDIT_INVENTORY, BoundaryKind
from src.utils.paths import PROJECT_ROOT

SCRIPTS: tuple[tuple[str, str, str, int], ...] = (
    ("ball_detection", "review_dataset", "review", 8776),
    ("ball_detection", "inference_ui", "inference", 8777),
    ("court_detection", "review_dataset", "review", 8774),
    ("court_detection", "inference_ui", "inference", 8775),
)

_NON_HYDRA_VALIDATOR = "src.utils.configuration.paths.NonHydraPathBoundary.validate"


def _script_module(task: str, script: str) -> ModuleType:
    return importlib.import_module(f"src.tasks.{task}.scripts.{script}")


def _record_serve_calls(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        module,
        "serve",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


@pytest.mark.parametrize("task,script,mode,port", SCRIPTS)
def test_help_exits_without_serving(
    task: str,
    script: str,
    mode: str,
    port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module(task, script)
    calls = _record_serve_calls(monkeypatch, module)
    monkeypatch.setattr(sys, "argv", [script, "--help"])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 0
    assert calls == []


@pytest.mark.parametrize("task,script,mode,port", SCRIPTS)
def test_validated_defaults_reach_serve(
    task: str,
    script: str,
    mode: str,
    port: int,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module(task, script)
    calls = _record_serve_calls(monkeypatch, module)
    repository = tmp_path / "repo"
    (repository / "data").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [script, "--project-root", str(repository)])

    module.main()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (task, mode)
    assert kwargs["port"] == port
    values = kwargs["values"]
    project = repository.resolve()
    assert values == {
        "project_root": project,
        "data_root": project / "data",
        "outputs_root": project / "outputs" / task,
        "checkpoints_root": project / "ckpt" / task,
    }


@pytest.mark.parametrize("task,script,mode,port", SCRIPTS)
def test_explicit_path_options_override_defaults(
    task: str,
    script: str,
    mode: str,
    port: int,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module(task, script)
    calls = _record_serve_calls(monkeypatch, module)
    repository = tmp_path / "repo"
    (repository / "data").mkdir(parents=True)
    custom_data = tmp_path / "custom-data"
    custom_data.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script,
            "--project-root",
            str(repository),
            "--data-root",
            str(custom_data),
            "--outputs-root",
            str(tmp_path / "out"),
            "--checkpoints-root",
            str(tmp_path / "ckpt"),
        ],
    )

    module.main()

    assert len(calls) == 1
    values = calls[0][1]["values"]
    assert values["data_root"] == custom_data.resolve()
    assert values["outputs_root"] == (tmp_path / "out").resolve()
    assert values["checkpoints_root"] == (tmp_path / "ckpt").resolve()


@pytest.mark.parametrize("task,script,mode,port", SCRIPTS)
def test_invalid_root_is_rejected_before_serve(
    task: str,
    script: str,
    mode: str,
    port: int,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module(task, script)
    calls = _record_serve_calls(monkeypatch, module)
    monkeypatch.setattr(
        sys,
        "argv",
        [script, "--project-root", str(tmp_path / "missing")],
    )

    with pytest.raises(PathContractError):
        module.main()

    assert calls == []


def test_detection_scripts_bind_their_inventory_boundary() -> None:
    source_root = (PROJECT_ROOT / "src").resolve()
    discovered = {
        (boundary.module, boundary.callable_name): boundary
        for boundary in discover_runtime_boundaries(source_root)
    }
    inventory = {
        (boundary.module, boundary.callable_name): boundary
        for boundary in DEFAULT_AUDIT_INVENTORY.boundaries
    }

    for task, script, _mode, _port in SCRIPTS:
        key = (f"src.tasks.{task}.scripts.{script}", "main")
        source = discovered[key]
        declared = inventory[key]
        assert source.kind is BoundaryKind.ARGPARSE
        assert source.validator_key == f"{task}.{script}"
        assert source.validator_callable == _NON_HYDRA_VALIDATOR
        assert source.executable_module
        assert (
            source.kind,
            source.validator_key,
            source.validator_callable,
            source.executable_module,
        ) == (
            declared.kind,
            declared.validator_key,
            declared.validator_callable,
            declared.executable_module,
        )

    assert ("src.tasks.base.visualization.detection.cli", "serve") not in discovered
