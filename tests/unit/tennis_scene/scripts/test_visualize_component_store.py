"""The component gallery renders a declared store read-only, in declared component order."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tennis_scene.pipeline.contracts import STANDARD_COMPONENTS
from src.tennis_scene.scripts.visualize_component_store import RENDERERS, Review
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.video import VideoInfo
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
    monkeypatch.setattr("src.tennis_scene.review.snapshot.probe_video_info", lambda _: VideoInfo(30., 1280, 720, 24))
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
    assert index.read_text().count("<section") == len(names) + 2
    snapshot = json.loads((tmp_path / "gallery/review.json").read_text())
    assert snapshot["export"]["status"] == "verified"
    assert snapshot["scene"]["arrays"]["ball_3d_valid"] == [True] * 24
    assert snapshot["sources"][0]["checksum_verified"]


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


def test_gallery_renderers_keep_gaps_on_the_source_frame_axis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review = Review.__new__(Review)
    review.output, review.frame_count = tmp_path, 5
    captured: dict[str, list[np.ndarray]] = {}
    def capture(path: Path, draw: Callable[[Any], None], **kwargs: Any) -> str:
        fig, ax = plt.subplots()
        try:
            draw(ax)
            captured[path.name] = [np.asarray(line.get_xydata()) for line in ax.lines]
        finally:
            plt.close(fig)
        return path.name
    monkeypatch.setattr("src.tennis_scene.scripts.visualize_component_store._save_plot", capture)
    valid = np.array([True, True, False, True, False])
    position = np.array([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.], [3., 3., 3.], [0., 0., 0.]])
    review.render_ball_triangulation("ball_triangulation", "", {"ball": {"status": "partial", "trajectory": {
        "positions": position, "valid": valid, "inliers": np.tile(valid, (3, 1)),
    }}})
    review.render_body_placement("body_placement", "", {"players": {"position": position[None], "yaw": np.zeros((1, 5)),
        "root_valid": valid[None], "heading_valid": valid[None], "smpl_valid": np.zeros((1, 5), bool), "vertices_local": None}})
    review.render_scene_assembly("scene_assembly", "", {"player_position": position[None], "player_valid": valid[None],
        "ball_3d": position, "ball_3d_valid": valid, "metadata": {"status": "partial"}})
    height = captured["ball_triangulation_height.png"][0]
    assert height[:, 0].tolist() == list(range(5))
    assert np.isnan(height[~valid, 1]).all()
    for name in ("body_placement_topdown.png", "scene_assembly_topdown.png"):
        trajectory = captured[name][-1]
        assert trajectory.shape == (5, 2)
        assert np.isnan(trajectory[~valid]).all()
        assert trajectory[0].tolist() == [0., 0.]
