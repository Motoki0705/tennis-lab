"""Real checkpoint bytes and fake CPU models exercise publication boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.submodules import models
from src.submodules.models.dino import person_detector
from src.tennis_scene.dataset_pipeline import people
from src.utils.checksum import FileIntegrityError


@pytest.fixture
def observation(tmp_path, monkeypatch):
    paths = SimpleNamespace(
        dino_checkpoint=tmp_path / "dino",
        vitpose_checkpoint=tmp_path / "pose",
        dino_repository=tmp_path,
    )
    paths.dino_checkpoint.write_bytes(b"detector")
    paths.vitpose_checkpoint.write_bytes(b"pose")
    (tmp_path / "cam0.mp4").write_bytes(b"video")
    pins = dict.fromkeys(
        ("court", "dino", "vitpose", "plcs", "blcs", "dinov3"), "0" * 64
    )
    pins.update(
        dino=people.sha256(paths.dino_checkpoint),
        vitpose=people.sha256(paths.vitpose_checkpoint),
    )
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "people": {
                "confidence": 0.5,
                "short_side": 32,
                "max_long_side": 64,
                "detection_stride": 1,
                "court_half_width_m": 6.5,
                "court_half_length_m": 18,
                "min_sample_coverage": 0.7,
                "max_gap_seconds": 2,
                "batch_size": 2,
                "precision": "float32",
            },
        }
    )
    monkeypatch.setattr(
        people,
        "read_clip",
        lambda *a, **kw: {
            "camera_ids": ["cam0"],
            "video_paths": ["cam0.mp4"],
            "num_frames": 2,
            "fps": 1,
        },
    )
    monkeypatch.setattr(
        people,
        "OpenCVVideoFrameReader",
        lambda *a, **kw: [
            SimpleNamespace(frame=np.zeros((32, 32, 3), np.uint8)) for _ in range(2)
        ],
    )
    detector = MagicMock()
    detector.predict.return_value = SimpleNamespace(
        boxes_xyxy=np.array([[-0.5, -14, 0.5, -12], [-0.5, 10, 0.5, 12]], np.float32),
        scores=np.ones(2, np.float32),
    )
    pose = MagicMock()
    pose.predict.return_value = SimpleNamespace(keypoints=torch.ones((2, 17, 3)))
    detector_factory = MagicMock(return_value=detector)
    pose_factory = MagicMock(return_value=pose)
    monkeypatch.setattr(person_detector, "DinoPersonDetector", detector_factory)
    monkeypatch.setattr(models, "ViTPosePose2D", pose_factory)
    monkeypatch.setattr(people.torch.cuda, "empty_cache", lambda: None)
    return SimpleNamespace(
        root=tmp_path,
        cfg=cfg,
        paths=paths,
        pins=pins,
        detector=detector,
        pose=pose,
        detector_factory=detector_factory,
        pose_factory=pose_factory,
    )


def observe(case, pins=True):
    people.observe_singles_people(
        case.cfg,
        case.paths,
        case.root,
        case.root,
        homographies=np.eye(3)[None],
        checkpoint_sha256=case.pins if pins else None,
    )


@pytest.mark.parametrize("pins", [False, True])
def test_new_observation_and_cache_reuse(observation, pins):
    case = observation
    observe(case, pins)
    before = {p.name: p.read_bytes() for p in case.root.glob("*.np*")}
    observe(case, pins)
    assert case.detector_factory.call_count == case.pose_factory.call_count == 1
    assert before == {p.name: p.read_bytes() for p in case.root.glob("*.np*")}
    # Raw cache reuse without a people cache also retains detector provenance.
    (case.root / "cam0_people.npz").unlink()
    (case.root / "cam0_people.metadata.json").unlink()
    observe(case, pins)
    assert case.detector_factory.call_count == 1
    assert case.pose_factory.call_count == 2


@pytest.mark.parametrize("role", ["dino", "vitpose"])
@pytest.mark.parametrize("cached", [False, True])
def test_agreeing_hash_providers_still_must_match_pin(
    observation, monkeypatch, role, cached
):
    case = observation
    if cached:
        observe(case)
    case.detector_factory.reset_mock()
    case.pose_factory.reset_mock()
    real_hash = people.sha256
    target = getattr(case.paths, f"{role}_checkpoint")
    monkeypatch.setattr(
        people, "sha256", lambda p: "f" * 64 if p == target else real_hash(p)
    )
    with pytest.raises(FileIntegrityError, match=f"mismatch for {role}"):
        observe(case)
    case.detector_factory.assert_not_called()
    case.pose_factory.assert_not_called()


@pytest.mark.parametrize("pins", [False, True])
@pytest.mark.parametrize("cached_people", [False, True])
def test_sibling_receipt_disagreement_is_fatal(observation, pins, cached_people):
    case = observation
    observe(case, pins)
    receipt = case.root / "cam0_detections.metadata.json"
    saved = json.loads(receipt.read_text())
    saved["checkpoint_sha256"] = "f" * 64
    receipt.write_text(json.dumps(saved))
    if not cached_people:
        (case.root / "cam0_people.npz").unlink()
        (case.root / "cam0_people.metadata.json").unlink()
    case.detector_factory.reset_mock()
    case.pose_factory.reset_mock()
    with pytest.raises(FileIntegrityError, match="checkpoint_sha256"):
        observe(case, pins)
    case.detector_factory.assert_not_called()
    case.pose_factory.assert_not_called()
    assert json.loads(receipt.read_text()) == saved


@pytest.mark.parametrize("pins", [False, True])
@pytest.mark.parametrize(
    "phase", ["detector_load", "detector_predict", "pose_load", "pose_predict"]
)
def test_changed_checkpoint_during_model_work_is_not_published(
    observation, pins, phase
):
    case = observation
    is_detector = phase.startswith("detector")
    target = (
        case.paths.dino_checkpoint if is_detector else case.paths.vitpose_checkpoint
    )
    model = case.detector if is_detector else case.pose
    boundary = (
        (case.detector_factory if is_detector else case.pose_factory)
        if phase.endswith("load")
        else model.predict
    )
    original = boundary.return_value

    def change(*args, **kwargs):
        target.write_bytes(b"changed during inference")
        return original

    boundary.side_effect = change
    with pytest.raises(FileIntegrityError, match="mismatch"):
        observe(case, pins)
    stem = "detections" if is_detector else "people"
    assert not (case.root / f"cam0_{stem}.npz").exists()
    assert not (case.root / f"cam0_{stem}.metadata.json").exists()
    assert not (case.root / "cam0_people.metadata.json").exists()


@pytest.mark.parametrize("pins", [False, True])
def test_raw_preflight_checks_pin_independently_of_people(
    observation, monkeypatch, pins
):
    case = observation
    real_hash = people.sha256
    reads = 0

    def changing_digest(path: Path) -> str:
        nonlocal reads
        if path == case.paths.dino_checkpoint:
            reads += 1
            if reads == 2:
                return "f" * 64
        actual: str = real_hash(path)
        return actual

    monkeypatch.setattr(people, "sha256", changing_digest)
    with pytest.raises(FileIntegrityError, match="mismatch for dino"):
        observe(case, pins)
    case.detector_factory.assert_not_called()
    case.pose_factory.assert_not_called()


@pytest.mark.parametrize("field", ["detector_sha256", "pose_sha256"])
def test_bad_people_receipt_stops_even_without_pins(observation, field):
    case = observation
    observe(case, False)
    receipt = case.root / "cam0_people.metadata.json"
    saved = json.loads(receipt.read_text())
    saved[field] = "f" * 64
    receipt.write_text(json.dumps(saved))
    with pytest.raises(FileIntegrityError, match=field):
        observe(case, False)
    assert json.loads(receipt.read_text()) == saved


def test_raw_receipt_changed_during_pose_is_rejected_before_publication(observation):
    case = observation
    original = case.pose.predict.return_value

    def corrupt_receipt(*args, **kwargs):
        receipt = case.root / "cam0_detections.metadata.json"
        saved = json.loads(receipt.read_text())
        saved["checkpoint_sha256"] = "f" * 64
        receipt.write_text(json.dumps(saved))
        return original

    case.pose.predict.side_effect = corrupt_receipt
    with pytest.raises(FileIntegrityError, match="checkpoint_sha256"):
        observe(case)
    assert not (case.root / "cam0_people.npz").exists()
    assert not (case.root / "cam0_people.metadata.json").exists()
