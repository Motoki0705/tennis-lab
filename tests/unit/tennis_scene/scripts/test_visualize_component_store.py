"""The component gallery renders a declared store read-only, in declared component order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.pipeline.contracts import STANDARD_COMPONENTS
from src.tennis_scene.scripts.visualize_component_store import RENDERERS, Review
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import setup_pipeline


def test_gallery_renders_every_declared_component_without_writing_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, _ = setup_pipeline(tmp_path, monkeypatch)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=tmp_path / "store")
    runner = pipeline.last_runner
    assert runner is not None
    # Every renderer understands exactly what its component declares.
    for name, node in runner.nodes.items():
        schema, version, _ = RENDERERS[name.split("/")[0]]
        assert (schema, version) == (node.io.output_schema, node.io.version), name
    assert set(RENDERERS) == set(STANDARD_COMPONENTS)

    monkeypatch.setattr(Review, "frame", lambda self, camera, index: np.zeros((720, 1280, 3), np.uint8))
    store = tmp_path / "store"
    before = {path: dual_sha256(path) for path in store.rglob("*") if path.is_file() and path.name != ".scene.lock"}
    index = Review(store / "scene.json", tmp_path / "gallery").build()
    after = {path: dual_sha256(path) for path in store.rglob("*") if path.is_file() and path.name != ".scene.lock"}
    assert before == after

    manifest = json.loads((tmp_path / "gallery/manifest.json").read_text())
    names = list(manifest["components"])
    assert [n.split("/")[0] for n in names] == sorted((n.split("/")[0] for n in names), key=STANDARD_COMPONENTS.index)
    assert {entry["status"] for entry in manifest["components"].values()} == {"rendered"}
    assert manifest["components"]["player_association"]["details"]["origin"] == "component"
    assert "player 0" in manifest["components"]["person_tracking/cam0"]["details"]["track 0 player"]
    assert index.read_text().count("<section") == len(names)


def test_gallery_is_written_outside_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, _ = setup_pipeline(tmp_path, monkeypatch)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=tmp_path / "store")
    with pytest.raises(ValueError, match="outside the component store"):
        Review(tmp_path / "store/scene.json", tmp_path / "store/gallery")
