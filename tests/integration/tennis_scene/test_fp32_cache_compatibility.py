"""Reopen a clip store written by main before the precision options existed."""

from __future__ import annotations

import json
import runpy
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.identity import PlayerAssociationModule
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingModule
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionModule
from src.tennis_scene.pipeline.components.pose_estimation import PoseEstimationModule
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    file_identity,
    standard_definition,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT

_BASELINE = "68b3bb4a491f6a7cca014c72d790a230a6a87894"
_BASELINE_STORE = Path(__file__).with_name("fixtures") / "main_fp32_cache.tar.gz"
_FIXTURE_HELPERS = "tests/unit/tennis_scene/pipeline/test_auto_pipeline.py"
_CAMERAS = ("cam0", "cam1", "cam2")
_CAMERA_STAGES = (CourtKPModule, BallDetectionModule, PersonDetectionModule, PersonTrackingModule,
                  PlayerSelectionModule, PoseEstimationModule)


def _portable_inputs(root: Path, patch: pytest.MonkeyPatch, fixture: dict[str, Any]) -> tuple[Path, ...]:
    # Keep the old store relocatable while hashing real fixture bytes.
    def relative_asset(path: Path) -> dict[str, Any]:
        identity: dict[str, Any] = file_identity(path)
        identity["path"] = path.relative_to(root).as_posix()
        return identity

    def relative_source(*args: Any, **kwargs: Any) -> Any:
        inputs = prepare_clip_source(*args, **kwargs)
        videos = tuple(replace(video, path=video.path.relative_to(root)) for video in inputs.source.videos)
        return replace(inputs, source=replace(inputs.source, videos=videos))

    patch.setattr("src.tennis_scene.pipeline.definition.file_identity", relative_asset)
    patch.setattr("src.tennis_scene.pipeline.orchestrator.file_identity", relative_asset)
    patch.setattr("src.tennis_scene.pipeline.orchestrator.prepare_clip_source", relative_source)
    paths: tuple[Path, ...] = fixture["patch_video_probe"](root, patch)
    return paths


def _write_baseline(root: Path, destination: Path) -> None:
    assert Path.cwd() == PROJECT_ROOT
    root = root.resolve()
    fixture = runpy.run_path(str(PROJECT_ROOT / _FIXTURE_HELPERS))
    cfg = fixture["runtime"](root, overrides=("device=cuda",))
    fixture["materialize_assets"](enabled_model_assets(cfg).values(), root)
    results = fixture["camera_stages"](*fixture["inputs"]())
    association = fixture["TrackIdentities"](_CAMERAS)

    def process(component: Any, request: Any) -> Any:
        return results[f"{component.io.name}/{request.video.camera_id}"].result

    with pytest.MonkeyPatch.context() as patch:
        for stage in _CAMERA_STAGES:
            patch.setattr(stage, "process", process)
        patch.setattr(PlayerAssociationModule, "process", lambda _, request: association.process(request))
        paths = _portable_inputs(root, patch, fixture)
        pipeline = TennisSceneOrchestrator(cfg)
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=_CAMERAS, store_root=root / "store")
        assert pipeline.last_runner is not None
        assert set(pipeline.last_runner.statuses.values()) == {"executed"}
        publication = pipeline.publication_identity()
    source = PROJECT_ROOT / "src"
    source_hashes = {str(path.relative_to(source)): dual_sha256(path) for path in sorted(source.rglob("*.py"))}
    assert document_digest(source_hashes) == publication["code_sha256"]
    (root / "baseline.json").write_text(json.dumps({
        "commit": _BASELINE, "publication": publication, "source_digests": source_hashes,
    }), encoding="utf-8")
    with tarfile.open(destination, "w:gz") as bundle:
        bundle.add(root / "baseline.json", arcname="baseline.json")
        bundle.add(root / "store/scene.json", arcname="store/scene.json")
        for directory in ("components", "exports"):
            bundle.add(root / "store" / directory, arcname=f"store/{directory}")


def _assert_cache_compatibility(root: Path, patch: pytest.MonkeyPatch, publication: dict[str, Any]) -> None:
    fixture = runpy.run_path(str(PROJECT_ROOT / _FIXTURE_HELPERS))
    paths = _portable_inputs(root, patch, fixture)
    cfg = fixture["runtime"](root, overrides=("device=cuda",))
    fixture["materialize_assets"](enabled_model_assets(cfg).values(), root)
    store_root = root / "store"
    index = json.loads((store_root / "scene.json").read_text(encoding="utf-8"))
    store = ClipStore(store_root, index["source"])
    legacy_refs = store.references()

    def no_prediction(*args: Any, **kwargs: Any) -> None:
        pytest.fail("FP32 is expected to adopt the main artifact from disk")

    for stage in (*_CAMERA_STAGES, PlayerAssociationModule):
        patch.setattr(stage, "process", no_prediction)

    for explicit in ((), _precision("fp32")):
        cfg = fixture["runtime"](root, overrides=("device=cuda", *explicit))
        pipeline = TennisSceneOrchestrator(cfg)
        assert pipeline.publication_identity() == publication
        for source, status in (("execute", "cached"), ("load", "loaded")):
            pipeline.config = replace(cfg, cache_source=source)
            pipeline.run(paths, video_role=PathRole.DATA, camera_ids=_CAMERAS, store_root=store_root)
            assert pipeline.last_runner is not None
            assert set(pipeline.last_runner.statuses.values()) == {status}
            assert pipeline.last_runner.references == legacy_refs

    assert pipeline.last_runner is not None
    clip = pipeline.last_runner.nodes["person_detection/cam0"].context.source
    affected = ("person_detection/cam0", "person_tracking/cam0", "pose_estimation/cam0")
    keys = {name: {legacy_refs[name].execution_key} for name in affected}
    for precision in ("bf16", "fp16"):
        cfg = fixture["runtime"](root, overrides=("device=cuda", *_precision(precision)))
        mixed = TennisSceneOrchestrator(cfg)
        assert mixed.publication_identity() != publication
        plan = ComponentRunner(standard_definition(cfg, clip, code_identity=mixed.code_identity), store).plan
        plan.references.update({(clip.clip_id, name): ref for name, ref in legacy_refs.items()})
        for name in affected:
            decision = plan.decide((clip.clip_id, name))
            assert decision.action == "execute"
            keys[name].add(document_digest(decision.identity))
    assert all(len(values) == 3 for values in keys.values())


def _precision(precision: str) -> tuple[str, ...]:
    return tuple(f"people_models.runtime.{model}.precision={precision}" for model in ("dino_detector", "vitpose"))


def test_main_fp32_cache_is_adopted_in_execute_and_load_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tennis_scene.pipeline import orchestrator, source_identity

    with tarfile.open(_BASELINE_STORE) as bundle:
        bundle.extractall(tmp_path, filter="data")
    saved = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert saved["commit"] == _BASELINE
    baseline = saved["source_digests"]
    assert document_digest(baseline) == saved["publication"]["code_sha256"]
    manifest = Path(source_identity.__file__).with_name("fp32_cache_compatibility.json")
    compatible = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    # Fix the hash inputs at the reviewed transition, independent of later source changes.
    snapshot = {**baseline, **{path: versions["after"] for path, versions in compatible.items()}}
    edited_patch = {**snapshot, "tennis_scene/pipeline/source_identity.py": "0" * 64}
    assert source_identity.compatible_source_digests(edited_patch) == edited_patch
    unrelated_edit = {"tennis_scene/schema.py": "0" * 64}
    assert source_identity.compatible_source_digests({**snapshot, **unrelated_edit}) == {**baseline, **unrelated_edit}
    monkeypatch.setattr(orchestrator, "source_digests", lambda _: snapshot)
    _assert_cache_compatibility(tmp_path, monkeypatch, saved["publication"])


if __name__ == "__main__":
    _write_baseline(Path(sys.argv[1]), Path(sys.argv[2]))
