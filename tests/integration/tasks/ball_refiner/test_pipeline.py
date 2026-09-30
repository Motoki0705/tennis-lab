"""Real CPU models/video -> immutable all-GMM artifacts -> fresh load-only runner."""

import json
import shutil
from dataclasses import fields, replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.deployment import (
    CENTRE_SELECTION,
    InferenceBundle,
    export_pilot_bundle,
    load_inference_bundle,
)
from src.tasks.ball_refiner.inference import predict_sequence
from src.tasks.ball_refiner.training.runner import run_training
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.ball_refiner_recipe import (
    RefinerEvidenceModule,
    ball_refiner_definition,
)
from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.ball_refiner import (
    BallRefiner2DInputAssembler,
    read_video_timeline,
)
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver
from tests.integration.tasks.ball_refiner.test_training import (
    config_for,
)
from tests.integration.tasks.ball_refiner.test_training import (
    pilot_inputs as pilot_inputs,
)
from tests.unit.tennis_scene.pipeline.config_factories import make_ball_config


@pytest.fixture(scope="module")
def exported(pilot_inputs, tmp_path_factory):
    root = tmp_path_factory.mktemp("refiner-pipeline")
    training = run_training(config_for(pilot_inputs, root / "train"))
    bundle = export_pilot_bundle(training, root / "bundle")
    best = json.loads((training / "best.json").read_text())
    expected = torch.load(training / best["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
    for key, value in bundle.load_model().model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    # Runtime must not need this directory (or the training store/cache).
    training.rename(root / "archived-training")
    return root, pilot_inputs, bundle


def _video(path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(13):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 10, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return SourceVideo("cam0", path, dual_sha256(path), 13, 30., 64, 48)


def _detector(root, pilot, bundle):
    base = make_ball_config(root)
    return replace(base, checkpoint=pilot / "detector.ckpt",
                   resolver=PathResolver(replace(base.resolver.roots, checkpoint_root=pilot)),
                   image_size=bundle.detector.image_size_hw, window_stride=bundle.detector.stride,
                   subpixel_refine=bundle.detector.subpixel_refine, candidates=bundle.detector.candidates,
                   overlap_aggregation=CENTRE_SELECTION, score_threshold=1.)


@pytest.fixture(scope="module")
def executed(exported):
    root, pilot, bundle = exported
    source = ClipSource("sample", (_video(root / "sample.mp4"),))
    config = _detector(root, pilot, bundle)
    nodes = ball_refiner_definition(source, detector_config=config, bundle_directory=bundle.directory,
                                    batch_size=2, code_identity="test", execution_source="execute")
    store = ClipStore(root / "store", json_value(source))
    runner = ComponentRunner(nodes, store)
    runner.run()
    return source, config, runner, bundle


def test_execute_and_fresh_load_preserve_full_gmm_without_inference(executed, monkeypatch):
    source, config, runner, bundle = executed
    output = runner.output("ball_refiner_2d/cam0")
    assert isinstance(output, BallRefiner2DOutput) and output.calibration == "uncalibrated"
    assert output.prediction.distribution.means.shape[:2] == (1, 13)
    detections = runner.output("ball_detection/cam0")
    assert not detections.observed.any()  # the point threshold rejects every frame
    assert detections.evidence.candidate_valid.any()  # pre-gate evidence survives
    assembled = BallRefiner2DInputAssembler(bundle).assemble(AssemblyContext(source, "cam0"), {"detections": detections})
    direct = predict_sequence(bundle.load_model(), assembled.model_input, window_length=bundle.window_length,
                              stride=bundle.stride, batch_size=2, device=torch.device("cpu"))
    for field in fields(direct.distribution):
        torch.testing.assert_close(getattr(output.prediction.distribution, field.name),
                                   getattr(direct.distribution, field.name), rtol=0, atol=0)
    np.testing.assert_array_equal(output.pts, assembled.pts)
    before = runner.store.index_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("load-only must not load a model, decode video or assemble inputs")

    monkeypatch.setattr(InferenceBundle, "load_model", forbidden)
    monkeypatch.setattr(RefinerEvidenceModule, "load", forbidden)
    monkeypatch.setattr(BallRefiner2DInputAssembler, "assemble", forbidden)
    nodes = ball_refiner_definition(source, detector_config=config, bundle_directory=bundle.directory,
                                    batch_size=2, code_identity="test", execution_source="load")
    fresh = ComponentRunner(nodes, ClipStore(runner.store.root, json_value(source), memory_entries=0))
    assert fresh.run() == runner.references
    assert set(fresh.statuses.values()) == {"loaded"}
    restored = fresh.output("ball_refiner_2d/cam0")
    for field in fields(output.prediction.distribution):
        torch.testing.assert_close(getattr(restored.prediction.distribution, field.name),
                                   getattr(output.prediction.distribution, field.name), rtol=0, atol=0)
    assert restored.camera_id == "cam0" and before == runner.store.index_path.read_bytes()
    px, covariance = restored.prediction.distribution.pixel_moments(torch.tensor([[64., 48.]]))
    assert px.shape == (1, 13, bundle.model_config.components, 2)
    assert (torch.linalg.eigvalsh(covariance) > 0).all()


@pytest.mark.parametrize("change", [
    {"overlap_aggregation": "max_score"}, {"window_stride": None}, {"subpixel_refine": False},
    {"normalize_imagenet": True}, {"image_size": (64, 64)}, {"tail_policy": "drop"}, {"checkpoint_strict": False},
])
def test_recipe_rejects_training_input_drift(exported, change):
    root, pilot, bundle = exported
    source = ClipSource("sample", (SourceVideo("cam0", root / "unused", "hash", 13, 30., 64, 48),))
    with pytest.raises(ValueError, match="frozen training evidence"):
        ball_refiner_definition(source, detector_config=replace(_detector(root, pilot, bundle), **change),
                                bundle_directory=bundle.directory, batch_size=2, code_identity="test", execution_source="execute")


def test_refiner_rejects_annotation_import_without_evidence(executed):
    source, _, runner, bundle = executed
    detection = runner.output("ball_detection/cam0")
    annotation = replace(detection, evidence=None, score_semantics="annotation_acceptance_not_probability")
    with pytest.raises(ValueError, match="no point fallback"):
        BallRefiner2DInputAssembler(bundle).assemble(AssemblyContext(source, "cam0"), {"detections": annotation})


@pytest.mark.parametrize("change", ["camera", "size", "window"])
def test_assembler_rejects_misaligned_evidence(executed, change):
    source, _, runner, bundle = executed
    detection = runner.output("ball_detection/cam0")
    if change == "camera":
        detection = replace(detection, camera_id="cam1")
    elif change == "size":
        old = detection.evidence
        evidence = replace(old, source_size_wh=(127, 95), candidate_uv_px=old.candidate_uv_px * 2)
        detection = replace(detection, evidence=evidence)
    else:
        old = detection.evidence
        start, slot = old.selected_window_start.copy(), old.selected_time_index.copy()
        start[2], slot[2] = 2, 0  # real source frame, but not the centre owner
        detection = replace(detection, evidence=replace(old, selected_window_start=start, selected_time_index=slot))
    with pytest.raises(ValueError, match="mismatch|training evidence policy"):
        BallRefiner2DInputAssembler(bundle).assemble(AssemblyContext(source, "cam0"), {"detections": detection})


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("num_frames", 14), ("width", 66), ("fps", 60.)])
def test_timeline_uses_source_pts_and_rejects_stale_metadata(executed, field, value):
    source = executed[0]
    with pytest.raises(ValueError):
        read_video_timeline(replace(source.videos[0], **{field: value}))


def test_saved_time_must_match_pts(executed):
    output = executed[2].output("ball_refiner_2d/cam0")
    expected = ((output.pts - output.pts[0]) * float(Fraction(output.time_base))).astype(np.float32)
    np.testing.assert_array_equal(output.timestamps_seconds, expected)
    with pytest.raises(ValueError, match="presentation timestamps"):
        replace(output, timestamps_seconds=output.timestamps_seconds + .1)


def test_corrupt_array_is_rejected_by_fresh_store(executed, tmp_path):
    source, _, runner, _ = executed
    shutil.copytree(runner.store.root, tmp_path / "corrupt")
    copied = ClipStore(tmp_path / "corrupt", json_value(source), memory_entries=0)
    ref = copied.active("ball_refiner_2d/cam0")
    descriptor = copied.descriptor(ref)
    array = copied.root / Path(ref.path).parent / next(iter(descriptor["arrays"]))
    array.write_bytes(array.read_bytes() + b"corrupt")
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

    with pytest.raises(ValueError, match="checksum"):
        copied.load(ref, ArtifactCodec(BallRefiner2DOutput))


def test_changed_bundle_does_not_restore_existing_artifact(executed, tmp_path):
    source, config, runner, bundle = executed
    shutil.copytree(bundle.directory, tmp_path / "bundle")
    path = tmp_path / "bundle" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stride"] = 1
    path.write_text(json.dumps(manifest))
    nodes = ball_refiner_definition(source, detector_config=config, bundle_directory=path.parent,
                                    batch_size=2, code_identity="test", execution_source="load")
    fresh = ComponentRunner(nodes, ClipStore(runner.store.root, json_value(source), memory_entries=0))
    with pytest.raises(ValueError, match="identity changed"):
        fresh.run()


def test_corrupt_bundle_is_rejected_before_model_load(exported, tmp_path):
    bundle = exported[2]
    shutil.copytree(bundle.directory, tmp_path / "bundle")
    weights = tmp_path / "bundle" / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_inference_bundle(weights.parent)


def test_export_rejects_overwrite_and_incomplete_training(exported, tmp_path):
    root, _, bundle = exported
    with pytest.raises(FileExistsError):
        export_pilot_bundle(root / "archived-training", bundle.directory)
    shutil.copytree(root / "archived-training", tmp_path / "train")
    state = tmp_path / "train" / "run_state.json"
    value = json.loads(state.read_text())
    value["status"] = "training"
    state.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completed pilot"):
        export_pilot_bundle(state.parent, tmp_path / "new-bundle")


@pytest.mark.parametrize("section,key", [("normalization", "enabled"), ("candidates", "patch_size")])
def test_bundle_never_fills_missing_detector_defaults(exported, tmp_path, section, key):
    shutil.copytree(exported[2].directory, tmp_path / "bundle")
    path = tmp_path / "bundle" / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["detector"][section][key]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every explicit field"):
        load_inference_bundle(path.parent)


def test_recipe_never_binds_another_camera(executed):
    source, config, _, bundle = executed
    source = replace(source, videos=(*source.videos, replace(source.videos[0], camera_id="cam1")))
    nodes = ball_refiner_definition(source, detector_config=config, bundle_directory=bundle.directory,
                                    batch_size=2, code_identity="test", execution_source="execute")
    assert len(nodes) == 4
    for node in nodes:
        if node.io.name == "ball_refiner_2d":
            assert node.bindings == {"detections": f"ball_detection/{node.context.camera_id}"}


def test_script_execute_and_load_without_training_store(exported, tmp_path, monkeypatch, capsys):
    from src.tasks.ball_detection.data.store import BallFrameStore
    from src.tasks.ball_refiner.scripts.run_pipeline import main

    _, pilot, bundle = exported
    video = _video(tmp_path / "cli.mp4")

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime must not read a training store")

    monkeypatch.setattr(BallFrameStore, "__init__", forbidden)
    args = ["run_pipeline", "--video", str(video.path), "--camera-id", "cam0",
            "--bundle", str(bundle.directory), "--detector-checkpoint", str(pilot / "detector.ckpt"),
            "--store", str(tmp_path / "store"), "--device", "cpu", "--detector-batch-size", "2",
            "--refiner-batch-size", "2", "--source", "execute"]
    monkeypatch.setattr("sys.argv", args)
    main()
    first = json.loads(capsys.readouterr().out.splitlines()[-1])
    monkeypatch.setattr(InferenceBundle, "load_model", forbidden)
    monkeypatch.setattr(RefinerEvidenceModule, "load", forbidden)
    monkeypatch.setattr("sys.argv", [*args[:-1], "load"])
    main()
    second = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert first["references"] == second["references"]
    assert set(first["components"].values()) == {"executed"}
    assert set(second["components"].values()) == {"loaded"}


@pytest.fixture
def calibrated_option(exported, tmp_path, monkeypatch):
    from src.tasks.ball_refiner import pipeline_options

    root, _, bundle = exported
    best = json.loads((root / "archived-training/best.json").read_text())
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema": "ball_refiner_2d.covariance_calibration.v1",
        "covariance_multiplier": 1.8125148752, "checkpoint_sha256": best["checkpoint_sha256"],
        "provenance": {"test": "fixed synthetic fixture"},
    }))
    identity = pipeline_options.BallPathIdentity(
        bundle.detector.checkpoint_sha256, best["checkpoint_sha256"], bundle.manifest_sha256, dual_sha256(path),
    )
    monkeypatch.setattr(pipeline_options, "E9_ANCHORED_S42", identity)
    return path, identity


def test_calibrated_option_execute_and_load_preserve_scaled_full_gmm(
    executed, calibrated_option, tmp_path, monkeypatch,
):
    from src.tennis_scene.pipeline.components.ball_refiner import (
        CalibratedBallRefiner2DOutput,
    )

    source, config, raw_runner, bundle = executed
    path, identity = calibrated_option
    options = dict(detector_config=config, bundle_directory=bundle.directory, batch_size=2,
                   code_identity="test", ball_path="e9_anchored_s42_covariance", calibration_artifact=path)
    nodes = ball_refiner_definition(source, execution_source="execute", **options)
    runner = ComponentRunner(nodes, ClipStore(tmp_path / "calibrated-store", json_value(source)))
    runner.run()
    output = runner.output("ball_refiner_2d/cam0")
    raw = raw_runner.output("ball_refiner_2d/cam0")
    assert isinstance(output, CalibratedBallRefiner2DOutput)
    assert output.calibration == "covariance_scale_v1"
    assert output.covariance_calibration.checkpoint_sha256 == identity.checkpoint_sha256
    assert output.calibration_artifact_sha256 == identity.calibration_sha256
    assert nodes[1].io.version == 2
    for name in ("means", "mixture_logits", "presence_logits"):
        torch.testing.assert_close(getattr(output.prediction.distribution, name),
                                   getattr(raw.prediction.distribution, name), atol=0, rtol=0)
    torch.testing.assert_close(output.prediction.distribution.covariance,
                               raw.prediction.distribution.covariance * 1.8125148752)
    descriptor = runner.store.descriptor(runner.references["ball_refiner_2d/cam0"])
    assert descriptor["identity"]["settings"]["covariance_calibration"]["artifact_sha256"] == dual_sha256(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("load must not construct models or assemble video inputs")

    monkeypatch.setattr(InferenceBundle, "load_model", forbidden)
    monkeypatch.setattr(RefinerEvidenceModule, "load", forbidden)
    monkeypatch.setattr(BallRefiner2DInputAssembler, "assemble", forbidden)
    fresh = ComponentRunner(ball_refiner_definition(source, execution_source="load", **options),
                            ClipStore(runner.store.root, json_value(source), memory_entries=0))
    assert fresh.run() == runner.references
    restored = fresh.output("ball_refiner_2d/cam0")
    assert isinstance(restored, CalibratedBallRefiner2DOutput)
    assert restored.covariance_calibration == output.covariance_calibration
    for field in fields(output.prediction.distribution):
        torch.testing.assert_close(getattr(restored.prediction.distribution, field.name),
                                   getattr(output.prediction.distribution, field.name), atol=0, rtol=0)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ball_refiner_definition(source, execution_source="load", **options)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "checkpoint", "bundle", "detector", "omitted"])
def test_named_option_stops_on_missing_or_wrong_assets(exported, calibrated_option, monkeypatch, failure):
    from src.tasks.ball_refiner import pipeline_options

    bundle = exported[2]
    path, identity = calibrated_option
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_text("{}")
    elif failure == "checkpoint":
        monkeypatch.setattr(pipeline_options, "E9_ANCHORED_S42", replace(identity, checkpoint_sha256="0" * 64))
    elif failure == "bundle":
        bundle = replace(bundle, manifest_sha256="0" * 64)
    elif failure == "detector":
        bundle = replace(bundle, detector=replace(bundle.detector, checkpoint_sha256="0" * 64))
    elif failure == "omitted":
        path = None
    with pytest.raises((ValueError, FileNotFoundError), match="missing|mismatch|requires"):
        pipeline_options.select_ball_path("e9_anchored_s42_covariance", bundle, path)


def test_no_implicit_calibration_or_unknown_ball_path(exported, calibrated_option):
    from src.tasks.ball_refiner.pipeline_options import select_ball_path

    bundle = exported[2]
    path = calibrated_option[0]
    assert select_ball_path("bundle", bundle, None) is None
    with pytest.raises(ValueError, match="explicitly select"):
        select_ball_path("bundle", bundle, path)
    with pytest.raises(ValueError, match="Unknown ball path"):
        select_ball_path("typo", bundle, path)
