"""Safety boundaries of the CPU rehearsal and generated GPU handoff commands."""

from __future__ import annotations

import hashlib
import importlib.util
import shlex
from pathlib import Path
from typing import Any

import pytest

from src.utils.paths import PROJECT_ROOT

_SPEC = importlib.util.spec_from_file_location("prepare_clean_install", PROJECT_ROOT / "scripts/prepare_clean_install.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def asset(filename: str, payload: bytes) -> dict[str, Any]:
    return {"filename": filename, "sha256": hashlib.sha256(payload).hexdigest(), "source": "official source", "local_acquisition": True}


def test_copy_checks_entire_selection_before_writing_and_never_replaces_files(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "good.pth").write_bytes(b"good")
    (source / "corrupt.pth").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _MODULE.copy_local_assets(source, target, [asset("good.pth", b"good"), asset("corrupt.pth", b"expected")])
    assert not target.exists()
    copied = _MODULE.copy_local_assets(source, target, [asset("good.pth", b"good")])
    assert copied[0]["sha256"] == hashlib.sha256(b"good").hexdigest()
    assert (source / "good.pth").stat().st_ino != (target / "good.pth").stat().st_ino
    with pytest.raises(ValueError, match="Conflicting"):
        _MODULE.copy_local_assets(source, target, [asset("good.pth", b"good")])


def test_copy_rejects_destination_escape_and_symlink(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "weights").mkdir()
    (source / "weights/model.pth").write_bytes(b"model")
    (target / "weights").symlink_to(source / "weights", target_is_directory=True)
    with pytest.raises(ValueError, match="escaping"):
        _MODULE.copy_local_assets(source, target, [asset("weights/model.pth", b"model")])
    with pytest.raises(ValueError, match="relative"):
        _MODULE.copy_local_assets(source, target, [asset("../model.pth", b"model")])


def test_generated_commands_quote_paths_and_use_only_fresh_venv(tmp_path: Path) -> None:
    repo = tmp_path / "space '$() tree" / "repo"
    build = repo / "outputs/dino_extension"
    pipeline, visualization = _MODULE.gpu_commands(repo, build)
    assert pipeline.startswith(f"cd {shlex.quote(str(repo))} && ")
    tokens = shlex.split(pipeline)
    assert str(repo / ".venv/bin/python") in tokens
    assert f"PYTHONPATH={build / 'lib'}" in tokens
    assert "CUDA_VISIBLE_DEVICES=''" not in pipeline
    assert "CUDA_VISIBLE_DEVICES=''" in visualization
    assert "src.tennis_scene.scripts.run_pipeline" in tokens
    assert "style.player_representation=smpl" in visualization
