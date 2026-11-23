"""Connect verified TensorRT assets to the standard scene declarations."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch

from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_model_components,
)
from src.tennis_scene.pipeline.model_resources import (
    clip_tensorrt,
    dino_tensorrt,
    gvhmr_tensorrt,
    hmr2_tensorrt,
    vitpose_tensorrt,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.tensorrt_assets import TensorRTModel
from src.utils.models.tensorrt_artifacts import (
    ENGINE_MANIFEST,
    EXPORT_MANIFEST,
    ONNX_FILE,
    PLAN_FILE,
    EngineEnvironment,
    EngineManifest,
    ExportManifest,
    InputProfile,
)
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _make_bundle(
    model: TensorRTModel, root: Path, environment: EngineEnvironment
) -> None:
    root.mkdir(parents=True)
    (root / ONNX_FILE).write_bytes(b"test export")
    (root / PLAN_FILE).write_bytes(b"test plan")
    export = ExportManifest(
        model=model.model,
        settings=model.settings,
        sources={name: _hash(path) for name, path in model.assets.items()},
        onnx_files={ONNX_FILE: _hash(root / ONNX_FILE)},
        inputs=model.inputs,
        outputs=model.outputs,
    )
    (root / EXPORT_MANIFEST).write_text(export.model_dump_json())
    profiles = {}
    for name, tensor in model.inputs.items():
        shapes = tuple(
            tuple(
                dim
                if dim != -1
                else batch * model.batch_multiplier
                if i == 0
                else frames
                for i, dim in enumerate(tensor.shape)
            )
            for batch, frames in ((1, 2), (2, 145), (4, 177))
        )
        profiles[name] = InputProfile(
            minimum=shapes[0], optimum=shapes[1], maximum=shapes[2]
        )
    engine = EngineManifest(
        export_sha256=_hash(root / EXPORT_MANIFEST),
        plan_sha256=_hash(root / PLAN_FILE),
        environment=environment,
        profiles=profiles,
    )
    (root / ENGINE_MANIFEST).write_text(engine.model_dump_json())


@pytest.mark.parametrize("legacy_gvhmr", [False, True])
def test_all_tensorrt_model_declarations_reach_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_gvhmr: bool,
) -> None:
    overrides = ("device=cuda", "gvhmr.enabled=true")
    base = runtime(tmp_path, overrides=overrides)
    materialize_assets(
        enabled_model_assets(base).values(), tmp_path, court=base.court_kp
    )
    fp32 = TennisSceneOrchestrator(base).publication_identity()
    people = base.people
    environment = EngineEnvironment(
        tensorrt_version="10.14.1", device_name="test GPU", compute_capability=(12, 0)
    )
    monkeypatch.setattr(
        "src.tennis_scene.pipeline.tensorrt_assets.resolve_device", torch.device
    )
    monkeypatch.setattr(
        "src.tennis_scene.pipeline.tensorrt_assets.tensorrt_module", object
    )
    monkeypatch.setattr(
        "src.tennis_scene.pipeline.tensorrt_assets.engine_environment",
        lambda *_: environment,
    )
    models = {
        "dino_detector": (
            "person_detection",
            "detector",
            dino_tensorrt(people, 750, 1333),
            2,
        ),
        "vitpose": ("person_pose_features", "pose", vitpose_tensorrt(people), 2),
        "clip_reid": (
            "person_appearance_features",
            "encoder",
            clip_tensorrt(base.tracking_encoder_weights),
            2,
        ),
        "hmr2": ("hmr2_features", "features", hmr2_tensorrt(people), 2),
        "gvhmr": ("gvhmr", "mesh", gvhmr_tensorrt(people, legacy=legacy_gvhmr), 1 if legacy_gvhmr else 4),
    }
    config_overrides = list(overrides)
    for name, (_, _, model, batch) in models.items():
        relative = f"tensorrt/{name}"
        _make_bundle(model, base.roots.artifact_root / relative, environment)
        prefix = f"people_models.runtime.{name}"
        config_overrides.extend(
            [
                f"{prefix}.backend=tensorrt",
                f"{prefix}.precision=fp16",
                f"{prefix}.engines=[{relative}]",
                f"{prefix}.batch_size={batch}",
            ]
        )
    cfg = runtime(tmp_path, overrides=tuple(config_overrides))
    configured = TennisSceneOrchestrator(cfg).publication_identity()
    components = standard_model_components(cfg, ())
    for _, (stage, alias, _, batch) in models.items():
        declaration = components[stage].model_specs()[alias]
        entry: dict[str, Any] = configured["models"][stage][alias]
        assert declaration.dtype == entry["dtype"] == "float16"
        assert declaration.settings == entry["settings"]
        assert entry["settings"]["backend"] == "tensorrt"
        assert entry["settings"]["precision"] == "fp16"
        assert entry["settings"]["batch_size"] == batch
        assert (
            entry["assets"]["engine_0/engine.plan"]["sha256"]
            == sha256(b"test plan").hexdigest()
        )
        assert fp32["models"][stage][alias]["dtype"] == "float32"
        assert "engine_0/engine.plan" not in fp32["models"][stage][alias]["assets"]
    too_large = runtime(tmp_path, overrides=(*config_overrides, "people_models.runtime.gvhmr.batch_size=5"))
    with pytest.raises(ValueError, match="outside profile"):
        standard_model_components(too_large, ())["gvhmr"].model_specs()

    # Four logical crops require eight engine images because flip is enabled.
    pose_batch = "people_models.runtime.vitpose.batch_size="
    config_overrides = [
        value for value in config_overrides if not value.startswith(pose_batch)
    ]
    too_large = runtime(tmp_path, overrides=(*config_overrides, f"{pose_batch}5"))
    with pytest.raises(ValueError, match=r"\(10, 3, 256, 192\) outside profile"):
        TennisSceneOrchestrator(too_large).publication_identity()
