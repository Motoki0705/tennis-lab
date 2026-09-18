"""Report failure semantics and loss-mask-aware trajectory statistics."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.tasks.slcs.data.quality import QualityConfig
from src.tennis_scene.dataset_pipeline.quality_report import (
    longest_gap,
    trajectory_metrics,
    write_quality_report,
)
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
    make_fixture_scene,
)


def test_weight_zero_outliers_and_gap_do_not_enter_supported_statistics() -> None:
    scene = make_fixture_scene(
        SLCSFixtureDatasetConfig(num_frames=5), np.random.default_rng(0)
    )
    scene.player_position[:] = 0
    scene.player_position[:, 2] = 1000
    assert scene.ball_3d is not None
    scene.ball_3d[:] = [0, 0, 1]
    scene.ball_3d[2] = [1000, 0, -100]
    mask = np.array([True, True, False, True, True])
    arrays = {
        "ball_reprojection_px": np.array([[1, 1, 999, 1, 1]]),
        "pose_reprojection_px": np.broadcast_to(
            np.array([1, 1, 999, 1, 1])[None, None, :, None], (2, 1, 5, 17)
        ),
    }
    supported = trajectory_metrics(scene, np.broadcast_to(mask, (2, 5)), mask, arrays)
    raw = trajectory_metrics(scene, np.ones((2, 5), bool), np.ones(5, bool), arrays)
    assert supported["player_speed_mps"]["count"] == 4
    assert supported["player_speed_mps"]["max"] == 0
    assert supported["ball_speed_mps"]["count"] == 2
    assert supported["ball_height_m"]["mean"] == 1
    assert supported["ball_negative_height_fraction_below_minus_0_1m"] == 0
    assert supported["pose_all_joint_reprojection_px"]["max"] == 1
    assert raw["ball_speed_mps"]["max"] > 1000
    assert raw["ball_negative_height_fraction_below_minus_0_1m"] == 0.2
    assert longest_gap(np.array([False, False, True, False])) == 2


def test_missing_clip_is_snapshot_not_completed_and_report_survives_failure(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    index = build_slcs_dataset_fixture(
        dataset, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    (
        index.clip_dir(index.clips[0]) / "annotations/tennis_scene/annotation.json"
    ).unlink()
    output = tmp_path / "report"
    arguments = dict(
        dataset=dataset,
        expected_dataset=dataset,
        runs=[],
        observations=[],
        output=output,
        quality=QualityConfig(0.3, 1, 1, 0.5),
    )
    with pytest.raises(ValueError, match="incomplete"):
        write_quality_report(**arguments)
    report = write_quality_report(**arguments, allow_incomplete=True)
    assert report["counts"]["missing"] == 1 and not report["complete"]
    assert (output / "quality_report.csv").is_file()


def test_corrupt_annotation_fails_even_in_snapshot(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    index = build_slcs_dataset_fixture(
        dataset, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    (
        index.clip_dir(index.clips[0]) / "annotations/tennis_scene/annotation.json"
    ).write_text("{}")
    with pytest.raises(ValueError, match="error"):
        write_quality_report(
            dataset,
            dataset,
            [],
            [],
            tmp_path / "report",
            quality=QualityConfig(0.3, 1, 1, 0.5),
            allow_incomplete=True,
        )
    report = json.loads((tmp_path / "report/quality_report.json").read_text())
    assert report["counts"]["error"] == 1


def test_mixed_provenance_cannot_be_full_success(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    build_slcs_dataset_fixture(
        dataset, SLCSFixtureDatasetConfig(videos=("video_000", "video_001"))
    )

    def result(*args: object, **kwargs: object) -> dict:
        key = str(args[1])
        return {
            "teacher_checkpoints": {"plcs": key},
            "teacher_settings_sha256": "x",
            "dino_checkpoint_sha256": "x",
            "dino_spec": {},
        }

    with (
        patch(
            "src.tennis_scene.dataset_pipeline.quality_report.audit_clip",
            side_effect=result,
        ),
        pytest.raises(ValueError, match="error"),
    ):
        write_quality_report(
            dataset,
            dataset,
            [],
            [],
            tmp_path / "report",
            quality=QualityConfig(0.3, 1, 1, 0.5),
            allow_incomplete=True,
        )
    report = json.loads((tmp_path / "report/quality_report.json").read_text())
    assert report["errors"] == ["Mixed provenance: teacher_checkpoints"]


def test_full_report_requires_every_expected_clip(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    build_slcs_dataset_fixture(dataset, SLCSFixtureDatasetConfig(videos=("video_000",)))
    result = {
        "teacher_checkpoints": {"plcs": "a" * 64},
        "teacher_settings_sha256": "x",
        "dino_checkpoint_sha256": "x",
        "dino_spec": {},
    }
    with patch(
        "src.tennis_scene.dataset_pipeline.quality_report.audit_clip",
        return_value=result,
    ):
        report = write_quality_report(
            dataset,
            dataset,
            [],
            [],
            tmp_path / "report",
            quality=QualityConfig(0.3, 1, 1, 0.5),
        )
    assert report["complete"] and report["status"] == "complete"
    assert report["counts"]["completed"] == report["expected_count"] == 1


def test_invalid_teacher_weight_is_a_contract_error(tmp_path: Path) -> None:
    from src.tennis_scene.dataset_pipeline.quality_report import audit_clip
    from src.tennis_scene.generate_dataset.manifest import ClipManifest

    dataset = tmp_path / "dataset"
    config = SLCSFixtureDatasetConfig(videos=("video_000",))
    index = build_slcs_dataset_fixture(dataset, config)
    clip = ClipManifest.load(index.clip_dir(index.clips[0]))
    scene = make_fixture_scene(config, np.random.default_rng(0))
    scene.metadata = {
        "reference": {"camera_ids": list(clip.camera_ids)},
        "dataset_producer_identity": {
            "checkpoints": {"plcs": "a" * 64, "blcs": "b" * 64}
        },
        "label_quality": {
            "schema_version": 1,
            "is_ground_truth": False,
            "player_weight": np.full((2, config.num_frames), -1).tolist(),
            "ball_weight": np.ones(config.num_frames).tolist(),
        },
    }
    with (
        patch(
            "src.tennis_scene.dataset_pipeline.quality_report.load_slcs_annotation",
            return_value=scene,
        ),
        pytest.raises(ValueError, match="teacher-quality weights"),
    ):
        audit_clip(
            clip,
            "video_000/clip_000",
            [],
            [],
            QualityConfig(0.3, 1, 1, 0.5),
            source_manifest_sha256=clip.digest(),
        )


def test_excluded_clip_still_present_in_dataset_is_error(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    build_slcs_dataset_fixture(dataset, SLCSFixtureDatasetConfig(videos=("video_000",)))
    with pytest.raises(ValueError, match="error"):
        write_quality_report(
            dataset,
            dataset,
            [],
            [],
            tmp_path / "report",
            quality=QualityConfig(0.3, 1, 1, 0.5),
            excluded_clips={"video_000/clip_000": "explicit exclusion"},
            allow_incomplete=True,
        )
    report = json.loads((tmp_path / "report/quality_report.json").read_text())
    assert "Excluded clips remain" in report["errors"][0]
    assert not report["complete"]


def test_input_identity_rejects_source_change_and_media_bitrot(tmp_path: Path) -> None:
    from src.tennis_scene.dataset_pipeline.quality_report import validate_input_identity
    from src.tennis_scene.generate_dataset.manifest import ClipManifest
    from src.tennis_scene.reference_pipeline.observations import sha256

    dataset = tmp_path / "dataset"
    index = build_slcs_dataset_fixture(
        dataset, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip = ClipManifest.load(index.clip_dir(index.clips[0]))
    identity = {
        "clip_manifest_sha256": clip.digest(),
        "video_sha256": {
            camera: sha256(clip.media_path(camera)) for camera in clip.camera_ids
        },
    }
    validate_input_identity(clip, identity, clip.digest())
    with pytest.raises(ValueError, match="source manifest"):
        validate_input_identity(clip, identity, "0" * 64)
    with clip.media_path(clip.camera_ids[0]).open("ab") as stream:
        stream.write(b"bitrot")
    with pytest.raises(ValueError, match="actual media bytes"):
        validate_input_identity(clip, identity, clip.digest())


def test_raw_checkpoint_and_observation_changes_are_rejected() -> None:
    import copy

    from src.tennis_scene.dataset_pipeline.quality_report import validate_raw_identity

    scene = make_fixture_scene(
        SLCSFixtureDatasetConfig(num_frames=3), np.random.default_rng(0)
    )
    scene.metadata = {
        "checkpoints": {"plcs": {"sha256": "a" * 64}},
        "reference": {},
        "ball_input_provenance": {},
    }
    identity = {"checkpoints": {"plcs": "a" * 64}}
    raw = copy.deepcopy(scene)
    validate_raw_identity(raw, scene, identity)
    raw.metadata["checkpoints"]["plcs"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_raw_identity(raw, scene, identity)
    raw = copy.deepcopy(scene)
    assert raw.ball_uv is not None
    raw.ball_uv[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="observation arrays"):
        validate_raw_identity(raw, scene, identity)
