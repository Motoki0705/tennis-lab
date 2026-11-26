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

from src.submodules.models.fasthmr.assets import CHECKPOINT_SHA256, FastHMRConfig
from src.submodules.models.fasthmr.inference import FastHMRPrediction, SelectionRows
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.body_types import NativeSMPLParameters
from src.tennis_scene.pipeline.clip_inputs import (
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodySelection,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.fasthmr import (
    FastHMRBody,
    FastHMROutput,
    NativeSMPLSegment,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    reconstruct_player_bodies,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.dataset_publication import publish_annotation
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    publication_model_identities,
    standard_definition,
)
from src.tennis_scene.pipeline.imports.publish import (
    BoundImport,
    bind_import,
    publish_import,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.source_identity import compatible_source_digests
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.video_batch import run_video_batch
from src.utils.checksum import dual_sha256
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


@pytest.fixture
def fast_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineRuntimeConfig:
    cfg: PipelineRuntimeConfig = runtime(tmp_path, overrides=FASTHMR)
    fast = cfg.fasthmr
    assert fast is not None
    source_code = fast.repository / "model.py"
    materialize_assets((*enabled_model_assets(cfg).values(), fast.demo_config, source_code), tmp_path, court=cfg.court_kp)
    monkeypatch.setattr(FastHMRConfig, "verify_repository", lambda _: {"source": source_code})
    for name, path in {**fast.inference_weights, **fast.geometry_weights}.items():
        if name in CHECKPOINT_SHA256:
            monkeypatch.setitem(CHECKPOINT_SHA256, name, dual_sha256(path))
    return cfg


@pytest.fixture
def selected_clip(tmp_path: Path) -> tuple[ClipSource, ClipStore]:
    source = ClipSource("short", (SourceVideo("cam0", tmp_path / "cam0.mp4", "a" * 64, 1, 30., 64, 36),))
    store = ClipStore(tmp_path / "fast", json_value(source))
    # One selected person with a single observation has no recoverable segment.
    selection = BodyViewSelectionOutput((BodySelection(0, "cam0", (), ()),))
    store.publish("body_view_selection", selection, ArtifactCodec(BodyViewSelectionOutput),
                  schema="body_view_selection", version=1, identity={"importer": "test-selection"},
                  dependencies={}, provenance={"origin": "import"})
    return source, store


def _recovery_nodes(cfg: PipelineRuntimeConfig, source: ClipSource) -> tuple[ComponentNode, ...]:
    nodes = {node.name: node for node in standard_definition(cfg, source, code_identity="test")}
    selection = nodes["body_view_selection"]
    # Start at the fixed selection to exercise the real FastHMR declaration and store.
    return replace(selection, io=replace(selection.io, inputs={}), bindings={}, source="load"), nodes["fasthmr"]


def test_fast_recipe_rejects_pt32_import_in_pt16_load(
    fast_config: PipelineRuntimeConfig, selected_clip: tuple[ClipSource, ClipStore],
) -> None:
    source, store = selected_clip
    runner = ComponentRunner(_recovery_nodes(fast_config, source), store)
    runner.run(["fasthmr"])
    pt32_output = runner.output("fasthmr")
    selection = store.references()["body_view_selection"]
    # Reproduce a pre-existing import without bypassing runner validation on read.
    imported = store.publish("fasthmr", pt32_output, ArtifactCodec(FastHMROutput),
                             schema="body_parameters", version=3,
                             identity={"importer": "test-import", "backend": "pt32"},
                             dependencies={"selection": selection}, provenance={"origin": "import"})
    assert fast_config.fasthmr is not None
    cfg = replace(fast_config, fasthmr=replace(fast_config.fasthmr, backend="pt16", device="cuda:1"),
                  component_sources={**fast_config.component_sources, "fasthmr": "load"})
    nodes = _recovery_nodes(cfg, source)
    load = ComponentRunner(nodes, store)
    with pytest.raises(ValueError, match="fasthmr requires component origin"):
        load.run(["fasthmr"])
    assert "fasthmr" not in load.references
    with pytest.raises(ValueError, match="fasthmr requires component origin"):
        bind_import(nodes, "fasthmr", store)
    bound = BoundImport(nodes[-1], {"selection": selection}, {})
    with pytest.raises(ValueError, match="fasthmr requires component origin"):
        publish_import(bound, store, pt32_output, importer="test-import", version=1,
                       identity={"backend": "pt32"}, provenance={})
    assert store.active("fasthmr") == imported


@pytest.mark.parametrize("entry", ["plan", "publication"])
def test_fast_recipe_rejects_changed_weights_even_without_requests(
    fast_config: PipelineRuntimeConfig, selected_clip: tuple[ClipSource, ClipStore], entry: str,
) -> None:
    source, store = selected_clip
    assert fast_config.fasthmr is not None
    fast_config.fasthmr.inference_weights["segmentor"].write_bytes(b"different segmentation checkpoint")
    with pytest.raises(ValueError, match="FastHMR official segmentor SHA-256 mismatch"):
        if entry == "plan":
            ComponentRunner(_recovery_nodes(fast_config, source), store).run(["fasthmr"])
        else:
            publication_model_identities(fast_config)
    assert store.active("fasthmr") is None


def test_only_the_exact_standard_recipe_patch_preserves_previous_source_identity() -> None:
    manifest = PROJECT_ROOT / "src/tennis_scene/pipeline/standard_recipe_cache_compatibility.json"
    records = json.loads(manifest.read_text())["files"]
    people = json.loads(manifest.with_name("ecpose_recipe_cache_compatibility.json").read_text())["files"]
    pose = json.loads(manifest.with_name("multiview_pose_cache_compatibility.json").read_text())["files"]
    comparison = json.loads(manifest.with_name("pose_comparison_cache_compatibility.json").read_text())["files"]
    plcs_path = "tennis_scene/pipeline/components/monocular_plcs.py"
    current = {
        path: dual_sha256(PROJECT_ROOT / "src" / path)
        for path in records.keys() | people.keys() | pose.keys() | comparison.keys() | {plcs_path}
    }
    main_sources = compatible_source_digests(current, tensorrt=True)
    assert {path: main_sources[path] for path in people} == {path: entry["after"] for path, entry in people.items()}
    outside_fast = {
        path: digest for path, digest in main_sources.items() if path not in records and path not in people
    }
    assert outside_fast[plcs_path] == current[plcs_path]
    outside_fast.update({
        path: entry["before"] for path, entry in people.items() if path not in records and entry["before"] is not None
    })
    fast_sources = compatible_source_digests(current, standard_people=True, tensorrt=True)
    assert fast_sources == {**outside_fast, **{path: entry["after"] for path, entry in records.items()}}
    baseline = {
        **outside_fast,
        **{path: entry["before"] for path, entry in records.items() if entry["before"] is not None},
    }
    assert compatible_source_digests(current, standard_recipe=True, tensorrt=True) == baseline
    local_edit = {**fast_sources, "tennis_scene/pipeline/definition.py": "a" * 64}
    assert compatible_source_digests(local_edit, standard_recipe=True, tensorrt=True) == local_edit
    people_edit = {"submodules/models/ecpose/runtime.py": "b" * 64}
    assert compatible_source_digests(current | people_edit, standard_recipe=True, tensorrt=True) == main_sources | people_edit


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_config: PipelineRuntimeConfig,
) -> None:
    legacy, _, stages = setup_pipeline(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset"
    clip = add_clip(dataset, "video_000/clip_000")
    materialize_assets(enabled_model_assets(legacy.config).values(), tmp_path, court=legacy.config.court_kp)
    standard = run_video_batch(dataset, "video_000", legacy, pipeline_config_yaml="recipe: standard\n")
    legacy_marker = standard.outcomes[0].annotation_path
    assert legacy_marker is not None
    legacy_bytes = legacy_marker.read_bytes()

    cfg = fast_config
    assert cfg.fasthmr is not None
    assert cfg.output_path.parent == tmp_path / "run/recipes/fasthmr"
    assert not cfg.people.hmr2_checkpoint.exists() and not cfg.people.gvhmr_checkpoint.exists()
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
    pt16_models = publication_model_identities(pt16)
    assert pt16_models["fasthmr"]["mesh"]["dtype"] == "float16"
    again = run_video_batch(dataset, "video_000", fast, pipeline_config_yaml="recipe: fasthmr\n")
    assert again.outcomes[0].status == "skipped"

    fast.config = replace(cfg, cache_source="load")
    loaded = fast.run(tuple(clip / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
                      camera_ids=("cam0", "cam1", "cam2"), store_root=legacy_marker.parent, clip_id="video_000/clip_000")
    assert fast.last_runner is not None and set(fast.last_runner.statuses.values()) == {"loaded"}
    assert calls == {"load": 1, "unload": 1, "predict": 1, "geometry_load": 1}
    marker_bytes = marker.read_bytes()
    record = select_dataset_clips(dataset, ("video_000/clip_000",))[0]
    with pytest.raises(ValueError, match="Published FastHMR differs from the declared execution settings"):
        publish_annotation(resolve_dataset_clip(dataset, record), loaded, pipeline_config_yaml="recipe: fasthmr\n",
                           publication_identity={**identity, "models": pt16_models})
    assert marker.read_bytes() == marker_bytes
    fast.config = replace(pt16, cache_source="load")
    with pytest.raises(ValueError, match="Loaded component identity changed: fasthmr"):
        fast.run(tuple(clip / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
                 camera_ids=("cam0", "cam1", "cam2"), store_root=legacy_marker.parent, clip_id="video_000/clip_000")
