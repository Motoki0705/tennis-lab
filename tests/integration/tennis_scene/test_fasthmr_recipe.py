"""Opt-in FastHMR selection, stored lineage and native geometry placement."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.submodules.models.fasthmr.assets import FastHMRConfig
from src.submodules.models.fasthmr.inference import FastHMRPrediction, SelectionRows
from src.tennis_scene.pipeline.body_types import NativeSMPLParameters
from src.tennis_scene.pipeline.components.fasthmr import (
    FastHMRBody,
    FastHMROutput,
    NativeSMPLSegment,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    reconstruct_player_bodies,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    publication_model_identities,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.source_identity import compatible_source_digests
from src.tennis_scene.pipeline.video_batch import run_video_batch
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT
from tests.support.tennis_scene.video_dataset import add_clip
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
    setup_pipeline,
)
from tests.unit.tennis_scene.pipeline.test_trace_player_placement import (
    _build_case,
    _FakeGeometry,
)

FASTHMR = (
    "recipe=fasthmr",
    "gvhmr.enabled=true",
    "+fasthmr={repository:FastHMR,assets:fasthmr,backend:pt32,feature_batch_size:2,seed:0}",
)


def test_only_the_exact_standard_recipe_patch_preserves_previous_source_identity() -> None:
    manifest = PROJECT_ROOT / "src/tennis_scene/pipeline/standard_recipe_cache_compatibility.json"
    records = json.loads(manifest.read_text())["files"]
    patches = {path: entry["after"] for path, entry in records.items()}
    baseline = {path: entry["before"] for path, entry in records.items() if entry["before"] is not None}
    assert compatible_source_digests(patches, standard_recipe=True, tensorrt=True) == baseline
    assert compatible_source_digests(patches, tensorrt=True) == patches
    local_edit = {**patches, "tennis_scene/pipeline/definition.py": "a" * 64}
    assert compatible_source_digests(local_edit, standard_recipe=True, tensorrt=True) == local_edit


@pytest.mark.parametrize("override,message", [
    ("recipe=typo", "Unknown pipeline recipe"),
    ("execution.gvhmr=load", "no execution.gvhmr stage"),
    ("fasthmr.backend=trt16", "TRT16"),
])
def test_recipe_rejects_unavailable_stage_or_backend(tmp_path: Path, override: str, message: str) -> None:
    options = (*FASTHMR[1:], override) if override.startswith("recipe=") else (*FASTHMR, override)
    with pytest.raises(ValueError, match=message):
        runtime(tmp_path, overrides=options)


def test_native_pose_drives_placement_with_legacy_scene_shape() -> None:
    case = _build_case(camera_quarter_turn=True)
    person = case.recovered.bodies[0]
    segments = tuple(
        NativeSMPLSegment(
            segment.source_frames,
            NativeSMPLParameters(
                np.pad(segment.parameters.body_pose, ((0, 0), (0, 6)), constant_values=.45),
                segment.parameters.global_orient, segment.parameters.betas, segment.parameters.transl,
            ),
            segment.observed_samples,
        )
        for segment in person.segments
    )
    result = reconstruct_player_bodies(
        case.grouped, case.raw, case.skeleton, case.cameras,
        FastHMROutput("fixture", (FastHMRBody(person.person_id, person.camera_id, segments, "fixture"),)),
        body=case.geometry, reprojection_px=20., placement_config=case.config,
    )
    assert result.diagnostics["body_model"] == "smpl" and result.body_pose.shape[-1] == 63
    assert result.smpl_valid[0, 0]
    np.testing.assert_allclose(result.position[0, 0], [1.65625, -1.6875, 1.40625], atol=1e-6)
    assert case.geometry.calls
    for parameters in case.geometry.calls:
        assert isinstance(parameters, NativeSMPLParameters)
        np.testing.assert_allclose(parameters.body_pose[:, -6:], .45, atol=1e-6)


def test_fast_recipe_publishes_and_reuses_native_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy, _, stages = setup_pipeline(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset"
    clip = add_clip(dataset, "video_000/clip_000")
    materialize_assets(enabled_model_assets(legacy.config).values(), tmp_path, court=legacy.config.court_kp)
    standard = run_video_batch(dataset, "video_000", legacy, pipeline_config_yaml="recipe: standard\n")
    legacy_marker = standard.outcomes[0].annotation_path
    assert legacy_marker is not None
    legacy_bytes = legacy_marker.read_bytes()

    cfg = runtime(tmp_path, overrides=FASTHMR)
    assert cfg.fasthmr is not None
    assert cfg.output_path.parent == tmp_path / "run/recipes/fasthmr"
    source_code = cfg.fasthmr.repository / "model.py"
    assets = enabled_model_assets(cfg)
    materialize_assets((*assets.values(), cfg.fasthmr.demo_config, source_code), tmp_path, court=cfg.court_kp)
    assert not cfg.people.hmr2_checkpoint.exists() and not cfg.people.gvhmr_checkpoint.exists()
    monkeypatch.setattr(FastHMRConfig, "verify_repository", lambda _: {"source": source_code})
    calls: Counter[str] = Counter()

    class Mesh:
        def __init__(self, _: FastHMRConfig) -> None:
            pass

        def load(self) -> None:
            calls["load"] += 1

        def unload(self) -> None:
            calls["unload"] += 1

        def predict(self, video: Path, size: tuple[int, int], rows: Sequence[SelectionRows], _: Any) -> tuple[FastHMRPrediction, ...]:
            calls["predict"] += 1
            assert video in {clip / f"cam{i}.mp4" for i in range(3)} and size == (1280, 720)
            return tuple(FastHMRPrediction(
                np.tile(np.eye(3, dtype=np.float32), (len(frames), 24, 1, 1)),
                np.zeros((len(frames), 10), np.float32), np.full((len(frames), 3), 1.2, np.float32),
            ) for frames, _ in rows)

    class Geometry(_FakeGeometry):
        def __init__(self, _: FastHMRConfig) -> None:
            super().__init__()

        def load(self) -> None:
            calls["geometry_load"] += 1

    monkeypatch.setattr("src.tennis_scene.pipeline.components.fasthmr.FastHMR", Mesh)
    monkeypatch.setattr("src.tennis_scene.pipeline.components.fasthmr.NativeSMPLGeometry", Geometry)
    fast = TennisSceneOrchestrator(cfg, components=stages)
    result = run_video_batch(dataset, "video_000", fast, pipeline_config_yaml="recipe: fasthmr\n")
    marker = result.outcomes[0].annotation_path
    assert marker == legacy_marker.parent / "recipes/fasthmr/annotation.json"
    assert legacy_marker.read_bytes() == legacy_bytes
    index = json.loads((marker.parent / "scene.json").read_text())
    assert "gvhmr" not in index["artifacts"] and "hmr2_features" not in index["artifacts"]
    assert index["artifacts"]["fasthmr"]["version"] == 3
    assert calls == {"load": 1, "unload": 1, "predict": 1, "geometry_load": 1}
    identity = fast.publication_identity()
    assert identity["recipe"] == "fasthmr" and identity["models"]["fasthmr"]["mesh"]["dtype"] == "float32"
    pt16 = replace(cfg, fasthmr=replace(cfg.fasthmr, backend="pt16"))
    assert publication_model_identities(pt16)["fasthmr"]["mesh"]["dtype"] == "float16"
    again = run_video_batch(dataset, "video_000", fast, pipeline_config_yaml="recipe: fasthmr\n")
    assert again.outcomes[0].status == "skipped"

    fast.config = replace(cfg, cache_source="load")
    fast.run(tuple(clip / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
             camera_ids=("cam0", "cam1", "cam2"), store_root=legacy_marker.parent, clip_id="video_000/clip_000")
    assert fast.last_runner is not None and set(fast.last_runner.statuses.values()) == {"loaded"}
    assert calls == {"load": 1, "unload": 1, "predict": 1, "geometry_load": 1}
    fast.config = replace(pt16, cache_source="load")
    with pytest.raises(ValueError, match="Loaded component identity changed: fasthmr"):
        fast.run(tuple(clip / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
                 camera_ids=("cam0", "cam1", "cam2"), store_root=legacy_marker.parent, clip_id="video_000/clip_000")
