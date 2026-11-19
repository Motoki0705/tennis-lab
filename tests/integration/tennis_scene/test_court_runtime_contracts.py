"""Court binding, cache cost and node provenance with CPU-only model fakes."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.court_kp import (
    CourtImagePrediction,
    CourtKPModule,
    CourtKPResult,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ComponentNode,
    ExecutionPlan,
)
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    CourtDetectionInputAssembler,
)
from src.tennis_scene.pipeline.model_resources import model_identities
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from tests.unit.tennis_scene.pipeline.components.test_court_resources import (
    external_court_pair,
)
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


class CachedCourtStore(ClipStore):
    """Only storage IO is fake; execution identities/decisions are production."""
    def __init__(self, root: Path, clip_id: str) -> None:
        self.root = root
        self.source_key = document_digest({"clip_id": clip_id})
        self.refs: dict[str, ArtifactRef] = {}
    def active(self, node: str) -> ArtifactRef | None:
        return self.refs.get(node)
    def load(self, reference: Any, codec: Any) -> CourtKPResult:
        return CourtKPResult(np.zeros((1, 1, 14, 2), np.float32), np.zeros((1, 1, 14), np.float32), np.array([0], np.int32))


def source(root: Path, clip: str, cameras: int) -> ClipSource:
    return ClipSource(clip, tuple(SourceVideo(f"cam{i}", root / f"{clip}-{i}.mp4", "media", 1, 30., 30, 20) for i in range(cameras)))


def cached_plan(cfg: Any, root: Path, clips: int, cameras: int, stores: list[CachedCourtStore]) -> ExecutionPlan:
    executions = []
    for i in range(clips):
        clip = f"clip{i}"
        nodes = standard_definition(cfg, source(root, clip, cameras), code_identity="fixed")
        if len(stores) <= i:
            stores.append(CachedCourtStore(root / clip, clip))
        executions.append(ClipExecution(clip, nodes, stores[i], targets=[f"court_detection/cam{c}" for c in range(cameras)]))
    return ExecutionPlan(executions)


def adopt_fake_cache(plan: ExecutionPlan) -> None:
    for key in plan.required:
        decision = plan.decide(key)
        store = plan.clips[key[0]].store
        assert isinstance(store, CachedCourtStore)
        store.refs[key[1]] = ArtifactRef("a" * 64, "court.json", "b" * 64, "court_observations", 2, document_digest(decision.identity))


def test_court_cache_cannot_cross_effective_device(tmp_path: Path) -> None:
    cfg = runtime(tmp_path, overrides=("person_observations.enabled=false", "player_reconstruction.enabled=false"))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    stores: list[CachedCourtStore] = []
    original = cached_plan(cfg, tmp_path, 1, 1, stores)
    adopt_fake_cache(original)
    same = cached_plan(cfg, tmp_path, 1, 1, stores).decide(("clip0", "court_detection/cam0"))
    assert same.action == "cached"
    changed = replace(cfg, court_kp=replace(cfg.court_kp, device="cuda"))
    decision = cached_plan(changed, tmp_path, 1, 1, stores).decide(("clip0", "court_detection/cam0"))
    assert decision.action == "execute"
    assert decision.identity["settings"]["model"]["dtype"] == "float32"


def test_dinov3_appearance_with_other_implementation_is_not_shared(tmp_path: Path) -> None:
    cfg = runtime(tmp_path)
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    appearance = replace(cfg.player_association.appearance, encoder="dinov3_vitb16")
    cfg = replace(cfg, player_association=replace(cfg.player_association, appearance=appearance))
    specs = []
    for label in ("a", "b"):
        root = tmp_path / label
        repo = root / "dinov3"
        repo.mkdir(parents=True)
        (repo / "hubconf.py").write_text(f"# implementation {label}\n")
        selected = replace(cfg, roots=replace(cfg.roots, external_asset_root=root))
        node = next(n for n in standard_definition(selected, source(tmp_path, label, 1), code_identity="fixed") if n.name == "player_association")
        spec = node.component.model_specs()["encoder"]
        specs.append(replace(spec, factory=lambda tag=label: SimpleNamespace(tag=tag)))
    a, b = ("a", "appearance"), ("b", "appearance")
    with InferenceRuntime({a: {"encoder": specs[0]}, b: {"encoder": specs[1]}}, cleanup=lambda _: None) as active:
        with active.node(a) as models:
            assert models.borrow("encoder").tag == "a"
        with active.node(b) as models:
            assert models.borrow("encoder").tag == "b"


def test_disabled_dinov3_appearance_does_not_require_its_repository(tmp_path: Path) -> None:
    cfg = runtime(tmp_path, overrides=("person_observations.enabled=false", "player_reconstruction.enabled=false"))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    cfg = replace(cfg, player_association=replace(cfg.player_association,
        appearance=replace(cfg.player_association.appearance, encoder="dinov3_vitb16")),
        roots=replace(cfg.roots, external_asset_root=tmp_path / "absent-appearance-code"))
    nodes = standard_definition(cfg, source(tmp_path, "disabled", 1), code_identity="fixed")
    association = next(node for node in nodes if node.name == "player_association")
    assert association.component.model_specs() == {}


@pytest.mark.parametrize("clips,cameras", [(1, 1), (1, 4), (69, 4)])
def test_all_cached_court_reads_metadata_and_hashes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: int, cameras: int) -> None:
    import src.tasks.court_detection.inference.checkpoint as checkpoint
    import src.tennis_scene.pipeline.inference_runtime as inference
    import src.tennis_scene.pipeline.model_assets as assets
    cfg = runtime(tmp_path, overrides=("person_observations.enabled=false", "player_reconstruction.enabled=false"))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    # Exceed the process-wide digest LRU: the planning catalog must retain
    # its checkpoint identity independently of the number of source files.
    repo = cfg.roots.external_asset_root / "dinov3"
    for i in range(130 if clips == 1 else 0):
        (repo / f"extra{i}.py").write_text(f"# source {i}\n")
    stores: list[CachedCourtStore] = []
    adopt_fake_cache(cached_plan(cfg, tmp_path, clips, cameras, stores))
    cfg = replace(cfg, court_kp=replace(cfg.court_kp))  # Fresh planning metadata context.
    assets._asset_digest.cache_clear()
    reads: list[Path] = []
    hashes: Counter[Path] = Counter()
    load, single, dual = torch.load, checkpoint.file_sha256, assets.dual_sha256
    def read(path: Any, *args: Any, **kwargs: Any) -> Any:
        reads.append(Path(path))
        return load(path, *args, **kwargs)
    def digest(path: Any) -> str:
        hashes[Path(path)] += 1
        return str(dual(path))
    def single_digest(path: Any) -> str:
        hashes[Path(path)] += 1
        return str(single(path))
    monkeypatch.setattr(torch, "load", read)
    monkeypatch.setattr(checkpoint, "file_sha256", single_digest)
    monkeypatch.setattr(assets, "dual_sha256", digest)
    monkeypatch.setattr(inference, "dual_sha256", digest)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Cached Court must not construct a model")
    monkeypatch.setattr(CourtKPModule, "_load_predictor", forbidden)
    plan = cached_plan(cfg, tmp_path, clips, cameras, stores)
    PlanExecutor(plan).run()
    assert set(plan.statuses.values()) == {"cached"}
    assert reads == [cfg.court_kp.checkpoint]
    assert hashes[cfg.court_kp.checkpoint] == 1
    assert hashes[cfg.court_kp.checkpoint] * cfg.court_kp.checkpoint.stat().st_size == cfg.court_kp.checkpoint.stat().st_size


@pytest.mark.parametrize("legacy", [False, True])
def test_shared_court_records_each_nodes_checkpoint_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool) -> None:
    import src.tennis_scene.pipeline.components.court_kp as component
    first, second, _, _ = external_court_pair(tmp_path, legacy=legacy)
    copy = second.config.resolver.roots.checkpoint_root / "copy.ckpt"
    copy.write_bytes(first.config.checkpoint.read_bytes())
    second = CourtKPModule(replace(second.config, checkpoint=copy))
    calls: list[Path] = []
    def loader(module: CourtKPModule) -> Any:
        calls.append(module.config.checkpoint)
        return SimpleNamespace(checkpoint_identity={"checkpoint_path": str(module.config.checkpoint),
            "backbone_asset_migration": {"checkpoint_root": str(module.config.resolver.roots.checkpoint_root)}})
    monkeypatch.setattr(CourtKPModule, "_load_predictor", loader)
    monkeypatch.setattr(component, "read_video_frame", lambda *args: SimpleNamespace(frame=np.zeros((20, 30, 3), np.uint8), original_size=(30, 20)))
    monkeypatch.setattr(CourtKPModule, "predict_image", lambda *args: CourtImagePrediction(np.full((14, 2), 5, np.float32), np.ones(14, bool), {}, None))
    def clip(clip_id: str, module: CourtKPModule, root: str) -> ClipExecution:
        bound_source = source(tmp_path, clip_id, 1)
        node = ComponentNode("court_detection/cam0", module, module.io, CourtDetectionInputAssembler(), {},
            AssemblyContext(bound_source, "cam0"), {"assets": module.asset_identities(),
                'models': model_identities(module.model_specs(), module.config.asset_catalog.file_identity)}, "fixed")
        return ClipExecution(clip_id, [node], ClipStore(tmp_path / root, {"clip_id": clip_id}, memory_entries=0))
    a, b = clip("A", first, "A"), clip("B", second, "B-shared")
    shared = PlanExecutor(ExecutionPlan([a, b]))
    shared.run()
    assert len(calls) == 1
    standalone = PlanExecutor(ExecutionPlan([clip("B", second, "B-alone")]))
    standalone.run()
    key = ("B", "court_detection/cam0")
    assert shared.output(key).diagnostics["checkpoint"]["checkpoint_path"] == str(copy)
    if legacy:
        assert shared.output(key).diagnostics["checkpoint"]["backbone_asset_migration"]["checkpoint_root"] == str(second.config.resolver.roots.checkpoint_root)
    assert json_value(shared.output(key)) == json_value(standalone.output(key))
    assert shared.plan.references[key].artifact_id == standalone.plan.references[key].artifact_id
