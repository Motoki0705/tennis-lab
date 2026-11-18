"""Cache/loader/publication boundaries use the same effective model declaration."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.tasks.player_association.appearance import encoders
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from src.tennis_scene.pipeline.model_assets import asset_hashes
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
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
    ("gvhmr", "device"), ("body_placement", "device"),
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
