"""Cache/loader/publication boundaries use the same effective model declaration."""

import weakref
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.tasks.player_association.appearance import encoders
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from src.tennis_scene.pipeline.model_assets import asset_hashes
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from tests.integration.tennis_scene.test_court_runtime_contracts import (
    CachedCourtStore,
    source,
)
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


def configured(tmp_path: Path) -> Any:
    cfg = runtime(tmp_path, overrides=("gvhmr.enabled=true",))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    return cfg


class InjectedCourt(CourtKPModule):
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
    def model_specs(self) -> Any:
        return {'court': ModelSpec('court', {'checkpoint': self.checkpoint}, 'tiny', 'cpu', 'float32', {}, object)}


def test_override_model_specs_separate_cache_and_publication(tmp_path: Path) -> None:
    cfg = configured(tmp_path)
    paths = [tmp_path / 'A.weights', tmp_path / 'B.weights']
    for path, content in zip(paths, [b'A', b'B'], strict=True):
        path.write_bytes(content)
    target = 'court_detection/cam0'
    store = CachedCourtStore(tmp_path / 'store', 'clip')
    markers, decisions = [], []
    for index, path in enumerate(paths):
        replacements = {target: InjectedCourt(path)}
        nodes = standard_definition(cfg, source(tmp_path, 'clip', 1), code_identity='fixed', overrides=replacements)
        plan = ExecutionPlan([ClipExecution('clip', nodes, store, targets=[target])])
        current = plan.decide(('clip', target))
        decisions.append(current)
        pipeline = TennisSceneOrchestrator(cfg, components=replacements)
        markers.append(pipeline.publication_identity())
        if index == 0:
            store.refs[target] = ArtifactRef('d' * 64, 'fake.json', 'e' * 64, current.identity['schema'],
                current.identity['version'], document_digest(current.identity))
    assert [d.action for d in decisions] == ['execute', 'execute']
    assert markers[0] != markers[1]
    assert paths[0] in asset_hashes(markers[0]) and paths[1] in asset_hashes(markers[1])


@pytest.mark.parametrize('boundary', ['node', 'publication'])
def test_unidentified_override_is_explicitly_rejected(tmp_path: Path, boundary: str) -> None:
    cfg = configured(tmp_path)
    class Unknown:
        io = CourtKPModule.io
        def process(self, inputs: Any) -> Any:
            raise AssertionError('Cannot execute an unidentified replacement')
    replacements = {'court_detection/cam0': Unknown()}
    with pytest.raises(ValueError, match='cache_identity'):
        if boundary == 'node':
            standard_definition(cfg, source(tmp_path, 'clip', 1), code_identity='fixed', overrides=replacements)
        else:
            TennisSceneOrchestrator(cfg, components=replacements).publication_identity()


@pytest.mark.parametrize('boundary', ['node', 'publication'])
def test_unknown_override_cannot_silently_use_the_standard_component(tmp_path: Path, boundary: str) -> None:
    cfg = configured(tmp_path)
    replacements = {'not_a_stage': InjectedCourt(cfg.court_kp.checkpoint)}
    with pytest.raises(ValueError, match='Unknown component override'):
        if boundary == 'node':
            standard_definition(cfg, source(tmp_path, 'clip', 1), code_identity='fixed', overrides=replacements)
        else:
            TennisSceneOrchestrator(cfg, components=replacements).publication_identity()


def decision(cfg: Any, root: Path, store: CachedCourtStore, target: str) -> Any:
    nodes = standard_definition(cfg, source(root, "clip", 1), code_identity="fixed")
    plan = ExecutionPlan([ClipExecution("clip", nodes, store, targets=[target])])
    # Hold dependencies fixed to check this node's own identity independently
    # of incidental invalidation through an upstream device change.
    plan.references.update({key: ArtifactRef("a" * 64, "fake.json", "b" * 64,
        node.io.output_schema, node.io.version, "c" * 64) for key, node in plan.nodes.items() if key[1] != target})
    return plan.decide(("clip", target))


@pytest.mark.parametrize("target,change", [
    ("ball_detection/cam0", "device"), ("ball_detection/cam0", "precision"),
    ("person_detection/cam0", "device"), ("person_tracking/cam0", "device"),
    ("player_selection/cam0", "device"), ("player_association", "device"),
    ("hmr2_features", "device"), ("gvhmr", "device"), ("body_placement", "device"),
])
def test_effective_model_settings_invalidate_own_cache(tmp_path: Path, target: str, change: str) -> None:
    cfg = configured(tmp_path)
    if change == "precision":
        from tests.unit.tennis_scene.pipeline.config_factories import (
            make_x3d_ball_config,
        )
        cfg = replace(cfg, ball_detection=make_x3d_ball_config(tmp_path))
        cfg.ball_detection.checkpoint.write_bytes(b"x3d checkpoint")
    store = CachedCourtStore(tmp_path / "store", "clip")
    first = decision(cfg, tmp_path, store, target)
    assert first.action == "execute"
    store.refs[target] = ArtifactRef("d" * 64, "fake.json", "e" * 64,
        first.identity["schema"], first.identity["version"], document_digest(first.identity))
    assert decision(cfg, tmp_path, store, target).action == "cached"
    if target.startswith("ball_detection"):
        changed = replace(cfg, ball_detection=replace(cfg.ball_detection, **({"device": "cuda"} if change == "device" else {"precision": "bf16"})))
        assert getattr(cfg.ball_detection, change) != getattr(changed.ball_detection, change)
    elif target.startswith("player_"):
        changed = replace(cfg, device="cuda")
    else:
        changed = replace(cfg, people=replace(cfg.people, runtime=replace(cfg.people.runtime, device="cuda")))
    assert decision(changed, tmp_path, store, target).action == "execute"


@pytest.mark.parametrize("consumer,name", [("person_tracking/cam0", "clipreid_vitb16_market1501"),
    ("player_association", "clipreid_vitb16_market1501"), ("player_association", "dinov3_vitb16")])
def test_appearance_factory_reads_the_declared_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str, name: str) -> None:
    cfg = configured(tmp_path)
    checkpoint = tmp_path / "ckpt" / "custom-appearance.pth"
    checkpoint.write_bytes(b"explicit custom checkpoint")
    cfg = replace(cfg, tracking_encoder_weights=checkpoint,
        association_encoder_weights=checkpoint,
        player_association=replace(cfg.player_association, appearance=replace(cfg.player_association.appearance, encoder=name)))
    calls: list[Path] = []
    def fake_clip(profile: str, path: Path, device: str) -> Any:
        calls.append(path)
    def fake_dino(profile: str, backbone: str, repo: Path, path: Path, device: str) -> Any:
        calls.append(path)
    monkeypatch.setattr(encoders, "ClipReIDEncoder", fake_clip)
    monkeypatch.setattr(encoders, "DINOv3Encoder", fake_dino)
    node = next(node for node in standard_definition(cfg, source(tmp_path, "clip", 1), code_identity="fixed") if node.name == consumer)
    spec = node.component.model_specs()["encoder"]
    spec.factory()
    assert calls == [spec.weights["checkpoint"]] == [checkpoint]


def test_publication_covers_every_enabled_node_model_asset(tmp_path: Path) -> None:
    cfg = configured(tmp_path)
    external = tmp_path / "appearance-code"
    (external / "dinov3").mkdir(parents=True)
    (external / "dinov3/hubconf.py").write_text("# separate appearance implementation\n")
    cfg = replace(cfg, roots=replace(cfg.roots, external_asset_root=external),
        player_association=replace(cfg.player_association, appearance=replace(cfg.player_association.appearance, encoder="dinov3_vitb16")))
    identity = TennisSceneOrchestrator(cfg).publication_identity()
    published = asset_hashes(identity)
    for node in standard_definition(cfg, source(tmp_path, "clip", 1), code_identity="fixed"):
        if hasattr(node.component, "model_specs"):
            for spec in node.component.model_specs().values():
                assert set(spec.weights.values()) <= published.keys(), node.name


def test_person_detector_implementation_is_bound_to_all_identities(tmp_path: Path) -> None:
    cfg = configured(tmp_path)
    path = cfg.people.dino_repository / "models/__init__.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# detector implementation A\n")
    util = cfg.people.dino_repository / "util/__init__.py"
    util.parent.mkdir(parents=True, exist_ok=True)
    util.write_text("# utilities\n")
    store = CachedCourtStore(tmp_path / "store", "clip")
    first = decision(cfg, tmp_path, store, "person_detection/cam0")
    spec_a = next(node for node in standard_definition(cfg, source(tmp_path, "clip", 1), code_identity="fixed")
        if node.name == "person_detection/cam0").component.model_specs()["detector"]
    key_a = spec_a.key
    marker = TennisSceneOrchestrator(cfg).publication_identity()
    path.write_text("# detector implementation B\n")
    spec_b = next(node for node in standard_definition(cfg, source(tmp_path, "clip", 1), code_identity="fixed")
        if node.name == "person_detection/cam0").component.model_specs()["detector"]
    assert "dino_source/models/__init__.py" in spec_a.weights
    assert key_a != spec_b.key
    assert first.identity != decision(cfg, tmp_path, store, "person_detection/cam0").identity
    assert marker != TennisSceneOrchestrator(cfg).publication_identity()
    # Specifications locate immutable asset inventories; current content is
    # checked at borrow, and missing implementations fail before model creation.
    assert spec_a.weights == spec_b.weights
    path.unlink()
    with pytest.raises(FileNotFoundError, match="DINO"):
        standard_definition(cfg, source(tmp_path, "clip", 1), code_identity="fixed")


@pytest.mark.parametrize("change", ["different", "removed"])
def test_native_dino_selection_is_bound_before_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    cfg = configured(tmp_path)
    extension = tmp_path / "selected.so"
    extension.write_bytes(b"native implementation A")
    monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(extension))
    node = next(node for node in standard_definition(cfg, source(tmp_path, "clip", 1), code_identity="fixed")
        if node.name == "person_detection/cam0")
    spec = node.component.model_specs()["detector"]
    assert extension in asset_hashes(node.settings)
    if change == "different":
        monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(tmp_path / "other.so"))
    else:
        monkeypatch.delenv("TENNIS_LAB_DINO_EXTENSION")
    initialized = []
    with pytest.raises(ValueError, match="extension selection changed"):
        spec.initialize(SimpleNamespace(load=lambda: initialized.append(True)))
    assert initialized == []


def person_runtime_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str) -> Any:
    """Real standard declaration/plan/store, replacing only DINO and decoded frames."""
    from src.submodules.models import PersonDetectionResult
    from src.tennis_scene.pipeline.components import person_detection as detector

    cfg = runtime(tmp_path, overrides=("device=cuda", f"people_models.runtime.dino_detector.precision={precision}"))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    extension = tmp_path / "pinned.so"
    extension.write_bytes(b"pinned native implementation A")
    monkeypatch.setenv("TENNIS_LAB_DINO_EXTENSION", str(extension))
    state = SimpleNamespace(events=[], factories=[], weakrefs=[], failure=None)

    class Detector:
        def __init__(self, checkpoint: Path, repository: Path, **options: Any) -> None:
            assert checkpoint == cfg.people.dino_checkpoint and repository == cfg.people.dino_repository
            state.factories.append(options)
            state.weakrefs.append(weakref.ref(self))
        def load(self) -> None:
            state.events.append("load")
            if state.failure == "initialize":
                raise RuntimeError("pinned DINO initialize failed")
        def unload(self) -> None:
            state.events.append("unload")
            if state.failure == "release":
                raise RuntimeError("pinned DINO release failed")
        def predict(self, request: Any) -> Any:
            state.events.append("predict")
            return PersonDetectionResult(np.array([[1, 2, 8, 14]], np.float32), np.array([.9], np.float32))

    monkeypatch.setattr(detector, "DinoPersonDetector", Detector)
    monkeypatch.setattr(detector, "OpenCVVideoFrameReader", lambda *a, **k:
        [SimpleNamespace(index=0, frame=np.zeros((20, 30, 3), np.uint8))])
    clip = source(tmp_path, "clip", 3)
    nodes = [node for node in standard_definition(cfg, clip, code_identity="fixed") if node.name.startswith("person_detection/")]
    return SimpleNamespace(cfg=cfg, extension=extension, clip=clip, nodes=nodes, state=state,
        store=ClipStore(tmp_path / "store", json_value(clip)))


@pytest.mark.parametrize("precision,dtype", [("fp32", "float32"), ("fp16", "float16")])
def test_standard_person_precision_and_native_pin_share_one_runtime_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str, dtype: str,
) -> None:
    case = person_runtime_case(tmp_path, monkeypatch, precision)
    nodes, clip, store, state = case.nodes, case.clip, case.store, case.state
    specs = [node.component.model_specs()["detector"] for node in nodes]
    assert len({spec.key for spec in specs}) == 1
    assert all(spec.dtype == dtype and spec.weights["extension"] == case.extension for spec in specs)
    marker = TennisSceneOrchestrator(case.cfg).publication_identity()
    assert asset_hashes(marker)[case.extension] == asset_hashes(nodes[0].settings)[case.extension]
    plan = ExecutionPlan([ClipExecution(clip.clip_id, nodes, store)])
    executor = PlanExecutor(plan)
    executor.run()
    assert len(state.factories) == 1 and state.factories[0]["precision"] == precision
    assert state.events == ["load", "predict", "predict", "predict", "unload"]
    assert all(ref() is None for ref in state.weakrefs)
    for node in nodes:
        output = executor.output((clip.clip_id, node.name))
        np.testing.assert_array_equal(output.boxes_xyxy, [[1, 2, 8, 14]])
        assert output.source_rows.tolist() == [0]
    state.events.clear()
    state.factories.clear()
    cached = ExecutionPlan([ClipExecution(clip.clip_id, nodes, store)])
    PlanExecutor(cached).run()
    assert set(cached.statuses.values()) == {"cached"}
    assert not state.factories and not state.events


def test_native_dino_pin_separates_runtime_cache_and_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = person_runtime_case(tmp_path, monkeypatch, "fp16")
    node = case.nodes[0]
    key = node.component.model_specs()["detector"].key
    PlanExecutor(ExecutionPlan([ClipExecution(case.clip.clip_id, [node], case.store)])).run()
    marker = TennisSceneOrchestrator(case.cfg).publication_identity()
    # Replace assets only between independent configurations, never within a plan.
    case.extension.write_bytes(b"pinned native implementation B")
    cfg = replace(case.cfg, court_kp=replace(case.cfg.court_kp))
    changed = next(n for n in standard_definition(cfg, case.clip, code_identity="fixed") if n.name == node.name)
    assert key != changed.component.model_specs()["detector"].key
    plan = ExecutionPlan([ClipExecution(case.clip.clip_id, [changed], case.store)])
    assert plan.decide((case.clip.clip_id, node.name)).action == "execute"
    assert marker != TennisSceneOrchestrator(cfg).publication_identity()


@pytest.mark.parametrize("failure", ["initialize", "release"])
def test_standard_fp16_person_failure_survives_measurement_suppression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    case = person_runtime_case(tmp_path, monkeypatch, "fp16")
    case.state.failure = failure
    node = case.nodes[0]
    plan = ExecutionPlan([ClipExecution(case.clip.clip_id, [node], case.store)])

    @contextmanager
    def suppressing_event(*args: Any, **kwargs: Any) -> Iterator[None]:
        with suppress(RuntimeError):
            yield

    completed: list[str] = []
    with pytest.raises(RuntimeError, match=f"pinned DINO {failure} failed") as caught:
        PlanExecutor(plan, model_event=suppressing_event).run(on_clip_complete=completed.append)
    assert caught.value.__traceback__ is not None  # Keep the exception while checking model references.
    assert case.state.events.count("load") == case.state.events.count("unload") == 1
    assert case.state.events.count("predict") == int(failure == "release")
    assert all(ref() is None for ref in case.state.weakrefs)
    assert not plan.references and not completed and case.store.active(node.name) is None
    assert plan.statuses[(case.clip.clip_id, node.name)] == "failed"
    restarted = ExecutionPlan([ClipExecution(case.clip.clip_id, [node], case.store)])
    assert restarted.decide((case.clip.clip_id, node.name)).action == "execute"
