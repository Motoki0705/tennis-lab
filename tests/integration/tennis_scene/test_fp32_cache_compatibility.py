"""Precision cache compatibility and the intentional C3 schema migration."""

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
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesModule,
    PersonPoseFeaturesModule,
)
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
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT

_BASELINE = "dcd6534594f73cca12a45c2d9afaa8ac09d48bc7"
_BASELINE_STORE = Path(__file__).with_name("fixtures") / "main_fp32_cache.tar.gz"
_FIXTURE_HELPERS = "tests/unit/tennis_scene/pipeline/test_auto_pipeline.py"
_CAMERAS = ("cam0", "cam1", "cam2")
_CAMERA_STAGES = (CourtKPModule, BallDetectionModule, PersonDetectionModule, PersonPoseFeaturesModule, PersonAppearanceFeaturesModule, PersonTrackingModule,
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


def _assert_cache_compatibility(root: Path, patch: pytest.MonkeyPatch) -> None:
    fixture = runpy.run_path(str(PROJECT_ROOT / _FIXTURE_HELPERS))
    paths = fixture["patch_video_probe"](root, patch)
    cfg = fixture["runtime"](root, overrides=("device=cuda",))
    fixture["materialize_assets"](enabled_model_assets(cfg).values(), root, court=cfg.court_kp)
    results = fixture["camera_stages"](*fixture["inputs"]())
    association = fixture["TrackIdentities"](_CAMERAS)

    def process(component: Any, request: Any, models: Any) -> Any:
        value = results[f"{component.io.name}/{request.video.camera_id}"].result
        if isinstance(component, PersonPoseFeaturesModule):
            return replace(value, model_identity=models.identity('pose'),
                           preprocessing_identity=document_digest(component.cache_identity()))
        if isinstance(component, PersonAppearanceFeaturesModule):
            return replace(value, encoder_identity=models.identity('encoder'),
                           preprocessing_identity=document_digest(component.preprocessing_settings()))
        return value

    for stage in _CAMERA_STAGES:
        if stage is PersonTrackingModule:
            patch.setattr(stage, "process", lambda component, request: results[f"{component.io.name}/{request.video.camera_id}"].result)
        else:
            patch.setattr(stage, "process_with_models", process)
    patch.setattr(PlayerAssociationModule, "process_with_models", lambda _, request, models: association.process(request))
    store_root = root / "store"
    pipeline = TennisSceneOrchestrator(cfg)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=_CAMERAS, store_root=store_root)
    assert pipeline.last_runner is not None
    assert set(pipeline.last_runner.statuses.values()) == {"executed"}
    publication = pipeline.publication_identity()
    store = pipeline.last_store
    assert store is not None
    legacy_refs = store.references()

    def no_prediction(*args: Any, **kwargs: Any) -> None:
        pytest.fail("FP32 is expected to adopt the main artifact from disk")

    for stage in (*_CAMERA_STAGES, PlayerAssociationModule):
        patch.setattr(stage, "process" if stage is PersonTrackingModule else "process_with_models", no_prediction)

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
    affected = ("person_detection/cam0", "person_pose_features/cam0")
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


def test_current_fp32_cache_is_adopted_in_execute_and_load_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_cache_compatibility(tmp_path, monkeypatch)


def test_exact_precision_patch_compatibility_does_not_hide_later_changes(tmp_path: Path) -> None:
    from src.tennis_scene.pipeline import source_identity

    with tarfile.open(_BASELINE_STORE) as bundle:
        bundle.extractall(tmp_path, filter="data")
    saved = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert saved["commit"] == _BASELINE
    baseline = saved["source_digests"]
    assert document_digest(baseline) == saved["publication"]["code_sha256"]
    manifest = Path(source_identity.__file__).with_name("fp32_cache_compatibility.json")
    compatible = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    snapshot = {**baseline, **{path: versions["after"] for path, versions in compatible.items()}}
    assert source_identity.compatible_source_digests(snapshot) == baseline
    edited_patch = {**snapshot, "tennis_scene/pipeline/source_identity.py": "0" * 64}
    assert source_identity.compatible_source_digests(edited_patch) == edited_patch
    unrelated_edit = {"tennis_scene/schema.py": "0" * 64}
    assert source_identity.compatible_source_digests({**snapshot, **unrelated_edit}) == {**baseline, **unrelated_edit}
    # The model runtime and tracking evidence schema are not precision-only changes.
    current = source_identity.source_digests(PROJECT_ROOT / "src")
    assert source_identity.compatible_source_digests(current, tensorrt=True) == current


def test_exact_tensorrt_patch_preserves_the_previous_fp32_source_identity() -> None:
    from src.tennis_scene.pipeline import source_identity

    manifest = Path(source_identity.__file__).with_name("tensorrt_fp32_cache_compatibility.json")
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    unrelated = {"tennis_scene/schema.py": "a" * 64}
    baseline = {**unrelated, **{path: record["before"] for path, record in files.items() if record["before"] is not None}}
    patch = {**unrelated, **{path: record["after"] for path, record in files.items()}}
    assert source_identity.compatible_source_digests(patch) == baseline
    assert source_identity.compatible_source_digests(patch, tensorrt=True) == patch
    changed = {**patch, "utils/models/tensorrt_runtime.py": "b" * 64}
    assert source_identity.compatible_source_digests(changed) == changed


def test_ball_side_preserves_the_exact_pre_player_source_identity() -> None:
    from src.tennis_scene.pipeline import source_identity

    manifest = Path(source_identity.__file__).with_name("ball_court_side_cache_compatibility.json")
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    unrelated = {"tennis_scene/schema.py": "a" * 64}
    baseline = {**unrelated, **{path: record["before"] for path, record in files.items() if record["before"] is not None}}
    patch = {**unrelated, **{path: record["after"] for path, record in files.items()}}
    assert document_digest(source_identity.compatible_source_digests(patch, ball_court_side=True)) == document_digest(baseline)
    assert source_identity.compatible_source_digests(patch) == patch  # Player uses its actual implementation.
    changed = {**patch, "tasks/court_side/hypothesis.py": "b" * 64}
    assert source_identity.compatible_source_digests(changed, ball_court_side=True) == changed


@pytest.mark.parametrize("name,old_version,new_version", [
    ("person_tracking/cam0", 5, 7), ("player_selection/cam0", 2, 3),
])
def test_pre_runtime_cache_requires_schema_regeneration(
    tmp_path: Path, name: str, old_version: int, new_version: int,
) -> None:
    from src.tennis_scene.pipeline.contracts import ClipSource
    from src.tennis_scene.pipeline.storage.codec import restore_type

    with tarfile.open(_BASELINE_STORE) as bundle:
        bundle.extractall(tmp_path, filter="data")
    index = json.loads((tmp_path / "store/scene.json").read_text(encoding="utf-8"))
    store = ClipStore(tmp_path / "store", index["source"])
    source = restore_type(index["source"], ClipSource)
    cfg = runpy.run_path(str(PROJECT_ROOT / _FIXTURE_HELPERS))["runtime"](tmp_path)
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import materialize_assets
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    nodes = standard_definition(cfg, source, code_identity="current-runtime")
    node = next(n for n in nodes if n.name == name)
    assert node.io.version == new_version
    assert store.references()[name].version == old_version
    plan = ComponentRunner(nodes, store).plan
    plan.references.update({(source.clip_id, key): ref for key, ref in store.references().items()})
    # Hold new feature dependencies fixed to isolate refusal of the legacy
    # tracking output. These references are never loaded or published.
    for dependency in plan.dependencies[(source.clip_id, name)].values():
        producer = plan.nodes[dependency]
        plan.references.setdefault(dependency, ArtifactRef('d' * 64, 'not-loaded.json', 'e' * 64,
            producer.io.output_schema, producer.io.version, 'f' * 64))
    assert plan.decide((source.clip_id, name)).action == "execute"
    load_nodes = [replace(n, source="load") for n in nodes]
    load = ComponentRunner(load_nodes, store).plan
    load.references.update(plan.references)
    with pytest.raises(ValueError, match="contract mismatch"):
        load.decide((source.clip_id, name))


if __name__ == "__main__":
    _write_baseline(Path(sys.argv[1]), Path(sys.argv[2]))
