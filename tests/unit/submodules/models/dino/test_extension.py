"""Importable legacy binaries must fail before loading the expensive model."""

import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch

from src.submodules.models import validate_dino_extension
from src.submodules.models.dino import architecture
from src.submodules.models.dino import extension as module
from src.utils.paths import PROJECT_ROOT


@pytest.fixture(autouse=True)
def isolate_loaded_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Loader state is process-lifetime in production; each test restores the
    # original state so fake modules cannot contaminate later real imports.
    monkeypatch.setattr(module, "_LOADED", {})
    monkeypatch.setattr(module, "_LOAD_ATTEMPTED", False)


@pytest.fixture
def binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = tmp_path / "MultiScaleDeformableAttention.so"
    path.write_text("marker = 'fixture'\n")
    monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(path))
    # Record an undo even if the key was absent before the loader inserts it.
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", ModuleType("placeholder"))
    monkeypatch.delitem(sys.modules, "MultiScaleDeformableAttention")
    original_spec = importlib.util.spec_from_file_location

    def source_spec(name: str, location: Path, **kwargs: Any) -> Any:
        if name == "MultiScaleDeformableAttention":
            kwargs["loader"] = importlib.machinery.SourceFileLoader(name, str(location))
        return original_spec(name, location, **kwargs)

    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", source_spec)
    return cast(ModuleType, module.load_dino_extension())


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
    assert validate_dino_extension() == Path(str(binary.__file__))
    assert [len(args) for args in calls] == [6, 7, 6, 7]


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


def test_preflight_preserves_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    error = ImportError("missing PyTorch symbol")
    path = tmp_path / "MultiScaleDeformableAttention.so"
    path.touch()
    monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(path))
    monkeypatch.delitem(sys.modules, "MultiScaleDeformableAttention", raising=False)

    def fail(spec: Any) -> None:
        raise error

    monkeypatch.setattr(module.importlib.util, "module_from_spec", fail)
    with pytest.raises(RuntimeError, match="Cannot import") as caught:
        validate_dino_extension()
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("selection", ["unset", "relative", "missing", "not_binary"])
def test_selection_fails_closed_with_a_cached_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    wrong = tmp_path / "MultiScaleDeformableAttention.py"
    wrong.write_text("raise AssertionError('shadow must not be imported')\n")
    monkeypatch.chdir(tmp_path)
    cached = ModuleType("MultiScaleDeformableAttention")
    cached.__file__ = str(wrong)
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", cached)
    if selection == "unset":
        monkeypatch.delenv("TENNIS_LAB_DINO_EXTENSION", raising=False)
    else:
        paths = {"relative": "MultiScaleDeformableAttention.so",
                 "missing": str(tmp_path / "missing.so"), "not_binary": str(wrong)}
        monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", paths[selection])
    with pytest.raises(RuntimeError, match="TENNIS_LAB_DINO_EXTENSION"):
        validate_dino_extension()


def test_different_cached_binary_is_rejected(binary: ModuleType, tmp_path: Path) -> None:
    binary.__file__ = str(tmp_path / "old.so")
    with pytest.raises(RuntimeError, match="already loaded from.*selects"):
        validate_dino_extension()


@pytest.mark.parametrize("foreign_path", ["selected", "other"])
def test_untracked_cached_module_cannot_be_retried_after_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_path: str
) -> None:
    selected = tmp_path / "MultiScaleDeformableAttention.so"
    selected.write_text("marker = 'selected'\n")
    foreign = ModuleType("MultiScaleDeformableAttention")
    foreign.__file__ = str(selected if foreign_path == "selected" else tmp_path / "other.so")
    monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(selected))
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", foreign)
    original_spec = importlib.util.spec_from_file_location

    def source_spec(name: str, location: Path) -> Any:
        return original_spec(name, location, loader=importlib.machinery.SourceFileLoader(name, str(location)))

    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", source_spec)
    with pytest.raises(RuntimeError, match="already loaded"):
        module.load_dino_extension()
    sys.modules.pop("MultiScaleDeformableAttention")
    replacement = tmp_path / "replacement"
    replacement.write_text("marker = 'replacement'\n")
    replacement.replace(selected)
    with pytest.raises(RuntimeError, match="already attempted.*fresh process"):
        module.load_dino_extension()


@pytest.mark.parametrize("change", ["atomic_replace", "in_place"])
def test_cached_binary_must_still_match_selected_file(binary: ModuleType, tmp_path: Path, change: str) -> None:
    def reject_cpu(*args: Any) -> None:
        raise RuntimeError("Not implemented on the CPU")

    binary.ms_deform_attn_forward = reject_cpu  # type: ignore[attr-defined]
    binary.ms_deform_attn_backward = reject_cpu  # type: ignore[attr-defined]
    selected = Path(str(binary.__file__))
    assert validate_dino_extension() == selected
    if change == "atomic_replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"different binary")
        replacement.replace(selected)
    else:
        selected.write_bytes(b"different binary")
    with pytest.raises(RuntimeError, match="changed.*fresh process"):
        validate_dino_extension()


def test_removed_cached_module_cannot_be_reimported(binary: ModuleType, tmp_path: Path) -> None:
    selected = Path(str(binary.__file__))
    replacement = tmp_path / "replacement"
    replacement.write_text("marker = 'replacement'\n")
    replacement.replace(selected)
    sys.modules.pop("MultiScaleDeformableAttention")
    with pytest.raises(RuntimeError, match="already attempted.*absent"):
        module.load_dino_extension()


def test_binary_changed_during_import_requires_fresh_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = tmp_path / "MultiScaleDeformableAttention.so"
    selected.write_text("marker = 'original'\n")
    monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(selected))
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", ModuleType("placeholder"))
    monkeypatch.delitem(sys.modules, "MultiScaleDeformableAttention")
    original_spec = importlib.util.spec_from_file_location

    class ReplacingLoader(importlib.machinery.SourceFileLoader):
        def exec_module(self, loaded: ModuleType) -> None:
            super().exec_module(loaded)
            replacement = tmp_path / "replacement"
            replacement.write_text("marker = 'replacement'\n")
            replacement.replace(selected)

    def source_spec(name: str, location: Path) -> Any:
        return original_spec(name, location, loader=ReplacingLoader(name, str(location)))

    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", source_spec)
    with pytest.raises(RuntimeError, match="changed during import"):
        module.load_dino_extension()
    assert "MultiScaleDeformableAttention" not in sys.modules
    with pytest.raises(RuntimeError, match="already attempted.*fresh process"):
        module.load_dino_extension()


@pytest.mark.parametrize("initially_cached", [False, True])
def test_explicit_file_loading_bypasses_cwd_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initially_cached: bool
) -> None:
    # Use the real import machinery with a source loader in place of the native
    # loader, so the portable test does not require a compiler or CUDA binary.
    selected = tmp_path / "MultiScaleDeformableAttention.so"
    selected.write_text(
        "marker = 'selected'\n"
        "def ms_deform_attn_forward(*args): raise RuntimeError('Not implemented on the CPU')\n"
        "def ms_deform_attn_backward(*args): raise RuntimeError('Not implemented on the CPU')\n"
    )
    shadow = tmp_path / "cwd"
    shadow.mkdir()
    (shadow / "MultiScaleDeformableAttention.py").write_text("raise AssertionError('shadow must not be imported')\n")
    name = "MultiScaleDeformableAttention"
    if initially_cached:
        original = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, original)
    else:
        original = None
        monkeypatch.delitem(sys.modules, name, raising=False)
    original_spec = importlib.util.spec_from_file_location

    def source_spec(name: str, location: Path) -> Any:
        return original_spec(name, location, loader=importlib.machinery.SourceFileLoader(name, str(location)))

    with monkeypatch.context() as isolated:
        isolated.chdir(shadow)
        isolated.syspath_prepend(str(shadow))
        isolated.setenv("TENNIS_LAB_DINO_EXTENSION", str(selected))
        # The loader inserts the module itself. Record an undo action even if
        # the name was originally absent, so its fake cannot leak to later tests.
        isolated.setitem(sys.modules, name, ModuleType(name))
        isolated.delitem(sys.modules, name)
        isolated.setattr(module.importlib.util, "spec_from_file_location", source_spec)
        assert validate_dino_extension() == selected
        assert sys.modules[name].marker == "selected"  # type: ignore[attr-defined]
    if initially_cached:
        assert sys.modules[name] is original
    else:
        assert name not in sys.modules


def test_runtime_import_uses_selected_module_and_rejects_cached_mismatch(
    binary: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "DINO"
    (repository / "util").mkdir(parents=True)
    (repository / "util/__init__.py").touch()
    (repository / "models").mkdir()
    (repository / "models/__init__.py").write_text(
        "import MultiScaleDeformableAttention as selected_extension\n"
        "def build_dino(args): return ()\n"
    )
    monkeypatch.setitem(sys.modules, "util", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, architecture._DINO_MODELS_PACKAGE, None)  # type: ignore[arg-type]
    # A prior real DINO import may leave this child after replacing its parent.
    # Isolate the fake repository and restore the prior function module on exit.
    monkeypatch.delitem(sys.modules, f"{architecture._DINO_MODELS_PACKAGE}.dino.ops.functions.ms_deform_attn_func", raising=False)
    build = architecture.load_dino_build_function(repository)
    upstream = sys.modules[build.__module__]
    assert upstream.selected_extension is binary  # type: ignore[attr-defined]
    binary.__file__ = str(tmp_path / "different.so")
    with pytest.raises(RuntimeError, match="already loaded from.*selects"):
        architecture.load_dino_build_function(repository)
    binary.__file__ = str(tmp_path / "MultiScaleDeformableAttention.so")
    stale = ModuleType("functions")
    stale.MSDA = ModuleType("old_extension")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, f"{architecture._DINO_MODELS_PACKAGE}.dino.ops.functions.ms_deform_attn_func", stale)
    with pytest.raises(RuntimeError, match="functions already reference a different extension"):
        architecture.load_dino_build_function(repository)


def test_build_rejects_existing_binary_before_build(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "lib").mkdir(parents=True)
    (output / "lib/MultiScaleDeformableAttention.so").touch()
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/build_dino_extension.sh"), str(tmp_path), str(output)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert not (output / "build.json").exists()
