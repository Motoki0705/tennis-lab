"""The component gallery renders a declared store read-only, in declared component order."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.pipeline.contracts import STANDARD_COMPONENTS
from src.tennis_scene.scripts.visualize_component_store import RENDERERS, Review
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from tests.support.tennis_scene.ball_refiner import (
    synthetic_model_recipe as synthetic_model_recipe,
)
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import setup_pipeline


@pytest.mark.parametrize("synthetic_model_recipe", [1, 2], indirect=True)
def test_gallery_renders_every_declared_component_without_writing_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, _ = setup_pipeline(tmp_path, monkeypatch)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=tmp_path / "store")
    runner = pipeline.last_runner
    assert runner is not None
    # Every renderer understands exactly what its component declares.
    for name, node in runner.nodes.items():
        schema, version, _ = RENDERERS[name.split("/")[0]]
        versions = (version,) if isinstance(version, int) else version
        assert schema == node.io.output_schema and node.io.version in versions, name
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
    assert "player 0" in manifest["components"]["player_selection/cam0"]["details"]["track 0 player"]
    assert index.read_text().count("<section") == len(names)


def test_gallery_is_written_outside_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, _ = setup_pipeline(tmp_path, monkeypatch)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=tmp_path / "store")
    with pytest.raises(ValueError, match="outside the component store"):
        Review(tmp_path / "store/scene.json", tmp_path / "store/gallery")


def test_gallery_keeps_legacy_filtered_point_history_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review = Review.__new__(Review)
    review.references = {"ball_points/cam0": {"version": 1}}
    def sheet(self: Review, node: str, camera: str, draw: Callable[[np.ndarray, int], None]) -> Path:
        for frame in range(2):
            draw(np.zeros((720, 1280, 3), np.uint8), frame)
        return tmp_path / "history.png"

    monkeypatch.setattr(Review, "sheet", sheet)
    monkeypatch.setattr(Review, "movie", lambda self, node, camera, draw: None)
    _, details = review.render_ball_points("ball_points/cam0", "cam0", {
        "uv_px": np.array([[100., 100.], [0., 0.]]), "observed": np.array([True, False]),
        "presence_probability": np.array([1., .1]), "area_px2": np.array([1., 40000.]),
        "rejection_codes": np.array([0, 3], np.uint8), "rule": {"min_presence": .9, "max_area_px2": 30000.},
    })
    assert dict(details)["accepted frames"] == "1"
    assert dict(details)["presence rejection"] == "1" and dict(details)["area rejection"] == "1"
    assert "historical confidence rule" in dict(details)
