"""Importable legacy binaries must fail before loading the expensive model."""

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from src.submodules.models import validate_dino_extension
from src.submodules.models.dino import extension as module


@pytest.fixture
def binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = tmp_path / "MultiScaleDeformableAttention.so"
    path.touch()
    binary = ModuleType("MultiScaleDeformableAttention")
    binary.__file__ = str(path)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: binary)
    return binary


def test_preflight_checks_forward_and_backward_on_cpu(binary: ModuleType) -> None:
    calls = []

    def reject_cpu(*args: Any) -> None:
        calls.append(args)
        assert all(tensor.device.type == "cpu" for tensor in args[:-1])
        assert args[1].dtype == args[2].dtype == torch.int64
        assert args[0].shape == (1, 1, 1, 1)
        assert args[-1] == 1
        raise RuntimeError("Not implemented on the CPU\nException raised from ms_deform_attn.h")

    binary.ms_deform_attn_forward = reject_cpu  # type: ignore[attr-defined]
    binary.ms_deform_attn_backward = reject_cpu  # type: ignore[attr-defined]
    assert validate_dino_extension() == Path(str(binary.__file__))
    assert [len(args) for args in calls] == [6, 7]
    assert calls[1][-2].shape == (1, 1, 1)


@pytest.mark.parametrize("entry", ["forward", "backward"])
@pytest.mark.parametrize("message", ["Undefined backend is not a valid device type", "Unrecognized tensor type ID: Undefined", "unrelated runtime error", ""])
def test_preflight_rejects_legacy_and_unrelated_errors(binary: ModuleType, entry: str, message: str) -> None:
    failure = RuntimeError(message)

    def supported(*args: Any) -> None:
        raise RuntimeError("Not implemented on the CPU")

    def incompatible(*args: Any) -> None:
        raise failure

    binary.ms_deform_attn_forward = supported  # type: ignore[attr-defined]
    binary.ms_deform_attn_backward = supported  # type: ignore[attr-defined]
    setattr(binary, f"ms_deform_attn_{entry}", incompatible)
    with pytest.raises(RuntimeError, match=f"failed CPU {entry} dispatch") as caught:
        validate_dino_extension()
    assert caught.value.__cause__ is failure
    assert "build_dino_extension.sh" in str(caught.value)


@pytest.mark.parametrize("behavior", ["missing", "accepts_cpu"])
def test_preflight_rejects_unexpected_extension(binary: ModuleType, behavior: str) -> None:
    if behavior == "accepts_cpu":
        binary.ms_deform_attn_forward = lambda *args: None  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="no forward entry|unexpectedly accepted CPU"):
        validate_dino_extension()


def test_preflight_preserves_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ImportError("missing PyTorch symbol")

    def fail(name: str) -> None:
        raise error

    monkeypatch.setattr(module.importlib, "import_module", fail)
    with pytest.raises(RuntimeError, match="Cannot import") as caught:
        validate_dino_extension()
    assert caught.value.__cause__ is error
