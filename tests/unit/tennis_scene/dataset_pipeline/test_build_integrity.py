"""Integrity failures stop the build while ordinary clip failures remain local."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.tennis_scene.dataset_pipeline import build
from src.utils.checksum import FileIntegrityError


@pytest.mark.parametrize("integrity", [True, False])
@pytest.mark.parametrize("preflight_mismatch", [True, False])
def test_build_records_failure_and_stops_only_for_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    integrity: bool,
    preflight_mismatch: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    runtime = SimpleNamespace(
        stage="all",
        seed=42,
        output=tmp_path / "output",
        source=tmp_path,
        destination=tmp_path / "destination",
        clip_ids=("first", "second"),
        dataset_clip_ids=None,
        features_enabled=True,
        feature_checkpoint=checkpoint,
        court=SimpleNamespace(checkpoint=checkpoint),
        checkpoint_sha256={
            name: "0" * 64
            for name in ("plcs", "blcs", "dino", "vitpose", "court", "dinov3")
        }
        if preflight_mismatch
        else None,
    )
    paths = SimpleNamespace(
        plcs_checkpoint=checkpoint,
        blcs_checkpoint=checkpoint,
        dino_checkpoint=checkpoint,
        vitpose_checkpoint=checkpoint,
    )
    monkeypatch.setattr(build.DatasetBuildConfig, "from_config", lambda cfg: runtime)
    monkeypatch.setattr(build.ReferenceClipPaths, "from_config", lambda cfg: paths)
    monkeypatch.setattr(build, "resolved_recipe", lambda cfg, runtime: {"stage": "all"})
    monkeypatch.setattr(
        build,
        "load_dataset_manifest",
        lambda path: SimpleNamespace(
            clips={name: SimpleNamespace(path=name) for name in runtime.clip_ids}
        ),
    )
    monkeypatch.setattr(build.ClipManifest, "load", lambda path: path)
    monkeypatch.setattr(build, "materialize_dataset", MagicMock())
    monkeypatch.setattr(build.torch.cuda, "empty_cache", MagicMock())
    features = MagicMock()
    monkeypatch.setattr(build, "_precompute_features", features)
    error = (
        FileIntegrityError("providers disagree", details={"path": "checkpoint"})
        if integrity
        else ValueError("quality failure")
    )
    process = MagicMock(side_effect=[error, None])
    monkeypatch.setattr(build, "_process_clip", process)
    if preflight_mismatch:
        with pytest.raises(RuntimeError, match="checkpoint"):
            build.build_dataset(OmegaConf.create({}))
        process.assert_not_called()
        features.assert_not_called()
        assert not runtime.output.exists()
        return
    with pytest.raises(FileIntegrityError if integrity else RuntimeError) as caught:
        build.build_dataset(OmegaConf.create({}))
    if integrity:
        assert caught.value is error
    assert process.call_count == (1 if integrity else 2)
    features.assert_not_called()
    failure = json.loads((runtime.output / "failures.json").read_text())
    assert failure == {
        "stage": "all",
        "failures": {"first": f"{type(error).__name__}: {error}"},
    }


def test_observe_stage_passes_runtime_pins_and_propagates_integrity(
    tmp_path, monkeypatch
):
    pins = {
        role: "a" * 64
        for role in ("court", "dino", "vitpose", "plcs", "blcs", "dinov3")
    }
    runtime = SimpleNamespace(
        stage="observe",
        observations=tmp_path / "observations",
        checkpoint_sha256=pins,
        resolver=SimpleNamespace(
            roots=SimpleNamespace(data_root=tmp_path, output_root=tmp_path)
        ),
    )
    clip = SimpleNamespace(clip_id="clip", clip_dir=tmp_path / "clip")
    paths = object()
    monkeypatch.setattr(build, "_import_ball", MagicMock())
    monkeypatch.setattr(
        build, "_observe_court", MagicMock(return_value=(None, "homographies"))
    )
    monkeypatch.setattr(build.ReferenceClipPaths, "from_config", lambda cfg: paths)
    error = FileIntegrityError("per-camera pin mismatch", details={})
    observe = MagicMock(side_effect=error)
    monkeypatch.setattr(build, "observe_singles_people", observe)
    with pytest.raises(FileIntegrityError) as caught:
        build._process_clip(OmegaConf.create({}), runtime, clip, tmp_path / "output")
    assert caught.value is error
    assert observe.call_args.kwargs["checkpoint_sha256"] is pins
    assert observe.call_args.kwargs["homographies"] == "homographies"


@pytest.mark.parametrize("pins_enabled", [False, True])
@pytest.mark.parametrize(
    "mismatch", [None, "dino", "vitpose", "sibling", "missing_raw"]
)
def test_infer_validates_receipts_before_scene_consumption(
    tmp_path, monkeypatch, pins_enabled, mismatch
):
    pins = dict.fromkeys(
        ("court", "dino", "vitpose", "plcs", "blcs", "dinov3"), "a" * 64
    )
    runtime = SimpleNamespace(
        stage="infer",
        observations=tmp_path / "observations",
        checkpoint_sha256=pins if pins_enabled else None,
        destination=tmp_path / "dataset",
        resolver=SimpleNamespace(
            roots=SimpleNamespace(data_root=tmp_path, output_root=tmp_path)
        ),
    )
    clip = SimpleNamespace(
        clip_id="clip", clip_dir=tmp_path / "clip", camera_ids=("cam0", "cam1")
    )
    observations = runtime.observations / clip.clip_id
    observations.mkdir(parents=True)
    for camera in clip.camera_ids:
        people = {"detector_sha256": pins["dino"], "pose_sha256": pins["vitpose"]}
        raw = {"checkpoint_sha256": pins["dino"]}
        if camera == "cam1":
            if mismatch == "dino":
                people["detector_sha256"] = raw["checkpoint_sha256"] = "f" * 64
            elif mismatch == "vitpose":
                people["pose_sha256"] = "f" * 64
            elif mismatch == "sibling":
                raw["checkpoint_sha256"] = "f" * 64
        (observations / f"{camera}_people.metadata.json").write_text(json.dumps(people))
        if mismatch != "missing_raw" or camera != "cam1":
            (observations / f"{camera}_detections.metadata.json").write_text(
                json.dumps(raw)
            )
    monkeypatch.setattr(build, "_import_ball", MagicMock())
    monkeypatch.setattr(build, "_observe_court", MagicMock(return_value=(None, None)))
    monkeypatch.setattr(build.ReferenceClipPaths, "from_config", lambda cfg: object())
    identity = MagicMock(return_value={"validated": True})
    monkeypatch.setattr(build, "scene_identity", identity)
    monkeypatch.setattr(
        build,
        "load_dataset_manifest",
        lambda path: SimpleNamespace(clips={"clip": SimpleNamespace(path="clip")}),
    )
    monkeypatch.setattr(build.ClipManifest, "load", lambda path: object())
    cached = MagicMock(return_value=False)
    monkeypatch.setattr(build, "validated_scene_cache", cached)
    runner = MagicMock(return_value=object())
    monkeypatch.setattr(build, "_scene_runner", runner)
    publish = MagicMock(return_value=[SimpleNamespace(status="published")])
    monkeypatch.setattr(build, "generate_pseudo_annotations", publish)
    observe = MagicMock(
        side_effect=AssertionError("infer must not load observation models")
    )
    monkeypatch.setattr(build, "observe_singles_people", observe)
    invalid = mismatch in {"sibling", "missing_raw"} or (
        pins_enabled and mismatch is not None
    )
    if invalid:
        with pytest.raises(FileIntegrityError):
            build._process_clip(
                OmegaConf.create({}), runtime, clip, tmp_path / "output"
            )
        identity.assert_not_called()
        cached.assert_not_called()
        runner.assert_not_called()
        publish.assert_not_called()
    else:
        build._process_clip(OmegaConf.create({}), runtime, clip, tmp_path / "output")
        publish.assert_called_once()
    observe.assert_not_called()
