from typing import Any, cast

import numpy as np
import pytest

from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline.people import select_court_halves


def _box(x, y, width, height):
    return np.array([x - width / 2, y - height, x + width / 2, y], np.float32)


def test_selects_both_ends_despite_two_larger_near_people():
    history: list[list[dict[str, Any]]] = [[] for _ in range(9)]
    for index in (0, 4, 8):
        history[index] = [
            {"id": 101, "bbx_xyxy": _box(0, -12, 1, 2)},
            {"id": 102, "bbx_xyxy": _box(-5.5, -12, 0.8, 1.8)},
            {"id": 103, "bbx_xyxy": _box(0, 12, 0.2, 0.4)},
            {"id": 104, "bbx_xyxy": _box(8, 12, 2, 4)},
        ]
    tracks, source_ids = select_court_halves(
        history,
        np.eye(3),
        half_width=5.8,
        half_length=18,
        sample_indices=np.array([0, 4, 8]),
        min_coverage=0.7,
        max_gap_frames=4,
    )
    assert tracks.track_ids == [0, 1]
    np.testing.assert_array_equal(source_ids[:, 0], [101, 103])
    np.testing.assert_array_equal(source_ids[:, 1], [-1, -1])
    assert tracks.observed_mask(0).sum() == 3
    assert (tracks.tracks[0][:, 3] < 0).all()
    assert (tracks.tracks[1][:, 3] > 0).all()


def test_refuses_missing_end_instead_of_interpolating_an_entire_player():
    history = [[{"id": 1, "bbx_xyxy": _box(0, -12, 1, 2)}] for _ in range(9)]
    with pytest.raises(ValueError, match="half 1 has insufficient"):
        select_court_halves(
            history,
            np.eye(3),
            half_width=5.8,
            half_length=18,
            sample_indices=np.arange(9),
            min_coverage=0.7,
            max_gap_frames=4,
        )


def test_refuses_long_unobserved_interval_despite_high_coverage():
    history: list[list[dict[str, Any]]] = [[] for _ in range(20)]
    for i in (0, 4, 16, 19):
        history[i] = [{"id": 1, "bbx_xyxy": _box(0, -12, 1, 2)}]
    with pytest.raises(ValueError, match="longest gap=11"):
        select_court_halves(
            history,
            np.eye(3),
            half_width=5.8,
            half_length=18,
            sample_indices=np.array([0, 4, 16, 19]),
            min_coverage=0.7,
            max_gap_frames=4,
        )


def test_mask_long_gaps_preserves_raw_detector_provenance():
    from src.tennis_scene.dataset_pipeline.people import pose_support_mask

    history: list[list[dict[str, Any]]] = [[] for _ in range(20)]
    indices = np.array([2, 5, 16, 18])
    for i in indices:
        history[i] = [
            {"id": 10, "bbx_xyxy": _box(0, -12, 1, 2)},
            {"id": 20, "bbx_xyxy": _box(0, 12, 1, 2)},
        ]
    tracks, ids = select_court_halves(
        history,
        np.eye(3),
        half_width=6.5,
        half_length=18,
        sample_indices=indices,
        min_coverage=0.7,
        max_gap_frames=2,
        long_gap_policy="mask",
        camera_id="cam2",
    )
    raw = tracks.observed_mask(0).numpy()
    np.testing.assert_array_equal(np.flatnonzero(raw), indices)
    supported = pose_support_mask(raw, 2)
    np.testing.assert_array_equal(np.flatnonzero(supported), [2, 3, 4, 5, 16, 17, 18])
    assert ids[0, 3] == -1


@pytest.mark.parametrize("samples,min_coverage", [([], 0), ([0], 0.5)])
def test_mask_still_rejects_missing_or_insufficient_track(samples, min_coverage):
    history: list[list[dict[str, Any]]] = [[] for _ in range(9)]
    for i in samples:
        history[i] = [{"id": 1, "bbx_xyxy": _box(0, -12, 1, 2)}]
    with pytest.raises(
        ValueError, match="cam2: Court half 0.*sample coverage=.*longest gap="
    ):
        select_court_halves(
            history,
            np.eye(3),
            half_width=6.5,
            half_length=18,
            sample_indices=np.arange(9),
            min_coverage=min_coverage,
            max_gap_frames=2,
            long_gap_policy="mask",
            camera_id="cam2",
        )


def test_association_masks_missing_view_and_ignores_unsupported_feet(tmp_path):
    from src.tennis_scene.reference_pipeline.reconstruction import associate_people

    for cam in ("cam0", "cam1", "cam2"):
        kp: np.ndarray = np.ones((2, 7, 17, 3), np.float32)
        kp[0, :, :, 1] = -12
        kp[1, :, :, 1] = 12
        supported: np.ndarray = np.ones((2, 7), bool)
        if cam == "cam2":
            supported[:, 1:6] = False
            kp[:, 1:6, :, 1] *= -1  # Would invert identity if included in median.
        np.savez(
            tmp_path / f"{cam}_people.npz",
            keypoints=kp,
            track_ids=np.array([10, 20]),
            pose_supported_mask=supported,
        )
    raw, assignments = associate_people(
        {"camera_ids": ["cam0", "cam1", "cam2"]},
        tmp_path,
        np.repeat(np.eye(3)[None], 3, axis=0),
        [False, False, True],
    )
    assert raw.shape == (2, 3, 7, 17, 3)
    assert (raw[:, :2, :, :, 2] == 1).all()
    assert (raw[:, 2, 1:6, :, 2] == 0).all()
    assert (raw[:, 2, [0, 6], :, 2] == 1).all()
    assert assignments["cam2"]["track_ids_near_far_cam0"] == [20, 10]
    assert assignments["cam2"]["median_ground_y_m"] == [-12, 12]


def test_association_ignores_zero_confidence_feet_in_legacy_cache(tmp_path):
    from src.tennis_scene.reference_pipeline.reconstruction import associate_people

    kp: np.ndarray = np.ones((2, 5, 17, 3), np.float32)
    kp[0, :, :, 1] = -12
    kp[1, :, :, 1] = 12
    kp[:, :3, :, 1] *= -1
    kp[:, :3, 15, 2] = 0
    np.savez(tmp_path / "cam0_people.npz", keypoints=kp, track_ids=np.array([0, 1]))
    _, assignments = associate_people(
        {"camera_ids": ["cam0"]}, tmp_path, np.eye(3)[None], [False]
    )
    assert assignments["cam0"]["track_ids_near_far_cam0"] == [0, 1]


def test_atomic_archive_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    from src.tennis_scene.dataset_pipeline.people import _save_npz_atomic

    cache = tmp_path / "detections.npz"

    def fail(handle, **arrays):
        handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        _save_npz_atomic(cache, boxes=np.empty((0, 4)))
    assert list(tmp_path.iterdir()) == []


def test_cache_settings_normalize_only_backward_compatible_error_policy():
    from omegaconf import OmegaConf

    from src.tennis_scene.dataset_pipeline.people import _people_cache_settings

    old = {"detection_stride": 4, "max_gap_seconds": 1.0}
    assert _people_cache_settings(OmegaConf.create(old)) == old
    assert (
        _people_cache_settings(OmegaConf.create({**old, "long_gap_policy": "error"}))
        == old
    )
    masked = {**old, "long_gap_policy": "mask"}
    assert _people_cache_settings(OmegaConf.create(masked)) == masked


@pytest.mark.parametrize("temporal", [False, True])
def test_observe_masks_pose_confidence_and_rejects_old_mask_cache(
    tmp_path, monkeypatch, temporal
):
    import json
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    from src.submodules import models
    from src.tennis_scene.dataset_pipeline import people

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "people": {
                "long_gap_policy": "mask",
                "court_half_width_m": 6.5,
                "court_half_length_m": 18,
                "min_sample_coverage": 0.7,
                "max_gap_seconds": 2,
                "batch_size": 2,
                "precision": "float32",
            },
        }
    )
    if temporal:
        cfg.people.selection_policy = "temporal_continuity"
        cfg.people.association = {"min_iou": 0.0, "max_center_distance": 1.0}
    # Only these two fields cross the mocked detector/pose boundary in this test.
    paths = cast(
        ReferenceClipPaths,
        SimpleNamespace(
            dino_checkpoint=tmp_path / "dino", vitpose_checkpoint=tmp_path / "pose"
        ),
    )
    clip = {
        "camera_ids": ["cam2"],
        "video_paths": ["cam2.mp4"],
        "num_frames": 10,
        "fps": 1,
    }
    indices = np.array([1, 3, 8])
    boxes = np.stack([_box(0, y, 1, 2) for _ in indices for y in (-12, 12)])
    monkeypatch.setattr(people, "read_clip", lambda *a, **kw: clip)
    monkeypatch.setattr(people, "sha256", lambda path: "fake-sha")
    monkeypatch.setattr(
        people,
        "_detections",
        lambda *a: (indices, np.array([0, 2, 4, 6]), boxes, np.ones(6)),
    )

    class FakePose:
        def __init__(self, *a, **kw):
            pass

        def predict(self, request):
            return SimpleNamespace(keypoints=torch.ones((10, 17, 3)))

        def unload(self):
            pass

    monkeypatch.setattr(models, "ViTPosePose2D", FakePose)
    people.observe_singles_people(
        cfg, paths, tmp_path, tmp_path, homographies=np.eye(3)[None]
    )
    cache = tmp_path / "cam2_people.npz"
    with np.load(cache) as saved:
        supported = saved["pose_supported_mask"]
        np.testing.assert_array_equal(np.flatnonzero(supported[0]), [1, 2, 3, 8])
        np.testing.assert_array_equal(
            np.flatnonzero(saved["observed_masks"][0]), indices
        )
        assert (saved["keypoints"][..., 2][~supported] == 0).all()
        assert (saved["keypoints"][..., 2][supported] == 1).all()
        assert (saved["keypoints"][..., :2] == 1).all()
    receipt = cache.with_suffix(".metadata.json")
    identity = json.loads(receipt.read_text())
    assert identity["schema_version"] == (4 if temporal else 3)
    assert identity["settings"]["long_gap_policy"] == "mask"
    people.observe_singles_people(
        cfg, paths, tmp_path, tmp_path, homographies=np.eye(3)[None]
    )
    identity["schema_version"] = 2
    receipt.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="Stale person observations"):
        people.observe_singles_people(
            cfg, paths, tmp_path, tmp_path, homographies=np.eye(3)[None]
        )


def test_zero_visibility_threshold_never_revives_masked_observations():
    from src.tennis_scene.reference_pipeline.reconstruction import (
        human_observation_mask,
    )

    uv = np.array([[0.5, 0.5], [0.5, 0.5], [1.2, 0.5]], np.float32)
    confidence = np.array([0.0, 0.2, 1.0], np.float32)
    np.testing.assert_array_equal(
        human_observation_mask(uv, confidence, 0.0), [False, True, False]
    )
    np.testing.assert_array_equal(
        human_observation_mask(uv, confidence, 0.3), [False, False, False]
    )


def test_heatmap_peaks_convert_explicitly_without_mutating_raw_observations():
    from src.tennis_scene.reference_pipeline.reconstruction import (
        pose_visibility_from_heatmap_peaks,
    )

    raw = np.array([-0.2, 0, 0.7, 1, 1.046875], np.float32)
    original = raw.copy()
    visibility, audit = pose_visibility_from_heatmap_peaks(raw)
    np.testing.assert_array_equal(visibility, np.array([0, 0, 0.7, 1, 1], np.float32))
    np.testing.assert_array_equal(raw, original)
    assert audit["method"] == "clip_heatmap_peak_to_unit_interval"
    assert audit["raw_min"] == float(raw.min())
    assert audit["raw_max"] == 1.046875
    assert audit["saturated_above_one_count"] == 1
    assert audit["saturated_below_zero_count"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_heatmap_peak_nonfinite_is_not_saturated(bad):
    from src.tennis_scene.reference_pipeline.reconstruction import (
        pose_visibility_from_heatmap_peaks,
    )

    with pytest.raises(ValueError, match="heatmap peaks.*finite"):
        pose_visibility_from_heatmap_peaks(np.array([0.0, bad]))
