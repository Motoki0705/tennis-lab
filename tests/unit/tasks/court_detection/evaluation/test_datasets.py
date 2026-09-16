"""Ground-truth loading and strata contracts for both benchmark domains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.tasks.court_detection.configuration import (
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.evaluation.datasets import (
    build_domain_records,
    load_sample,
    sample_ref,
)
from src.tasks.court_detection.evaluation.settings import BenchmarkDomainSettings
from src.tasks.court_detection.geometry.homography import court_template_xy
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    COURT_KP_NAMES,
)

_WIDTH = 64
_HEIGHT = 48
_CAMERA = {
    "camera_id": "sample-train",
    "source_frame_index": 0,
    "width": _WIDTH,
    "height": _HEIGHT,
    "intrinsics": [50.0, 0.0, 31.5, 0.0, 50.0, 23.5, 0.0, 0.0, 1.0],
    "camera_to_scene": [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    "image_path": "generated/court/sample-train.png",
}


def _target_court() -> dict[str, object]:
    return {
        "binding": {
            "court_instance_id": "court-a",
            "candidate_id": "candidate-a",
            "scene_from_court": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "selection_seed": 1,
        },
        "resolution_policy": "nearest_camera",
        "camera_to_court_center_distance_m": 9.0,
    }


def _projection(
    sample_id: str, *, physical_order: tuple[int, ...], coverage_mode: str
) -> dict[str, object]:
    classes = []
    for class_id, (class_name, physical_index) in enumerate(
        zip(COURT_KP_NAMES[:14], physical_order, strict=True)
    ):
        point = {
            "physical_index": physical_index,
            "uv": [float(6 + class_id * 2), float(6 + physical_index)],
            "camera_depth_m": 12.0,
            "scene_xyz_m": [float(physical_index), 0.0, 0.0],
            "in_front": True,
            "in_frame": True,
            "renderer_visible": True,
        }
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "renderer_visible": True,
                "points": [point],
            }
        )
    return {
        "camera_id": sample_id,
        "resolution": [_WIDTH, _HEIGHT],
        "coverage_modes": [coverage_mode],
        "visible_class_names": list(COURT_KP_NAMES[:14]),
        "visible_point_count": 14,
        "courts": [
            {
                "court_instance_id": "court-a",
                "coverage_mode": coverage_mode,
                "classes": classes,
            }
        ],
    }


def _write_synthetic_scene(
    workspace: Path,
    *,
    scene_id: str,
    physical_order: tuple[int, ...],
    coverage_mode: str,
    sample_id: str,
    split: str,
    group_id: str,
) -> None:
    dataset_root = workspace / scene_id / "datasets" / "court"
    relative = Path("samples") / split / group_id / sample_id
    sample_root = dataset_root / relative
    sample_root.mkdir(parents=True, exist_ok=True)
    rgb: NDArray[np.float32] = np.full((_HEIGHT, _WIDTH, 3), 0.5, dtype=np.float32)
    np.save(sample_root / "rgb.npy", rgb)
    np.save(sample_root / "alpha.npy", np.ones((_HEIGHT, _WIDTH, 1), np.float32))
    np.save(sample_root / "depth.npy", np.ones((_HEIGHT, _WIDTH, 1), np.float32))
    Image.fromarray((rgb * 255).astype(np.uint8)).save(sample_root / "rgb.png")
    Image.new("L", (_WIDTH, _HEIGHT)).save(sample_root / "alpha.png")
    projection = _projection(
        sample_id, physical_order=physical_order, coverage_mode=coverage_mode
    )
    camera = dict(_CAMERA)
    camera["camera_id"] = sample_id
    labels = {
        "schema": "canonical_court_sample_v3",
        "sample_index": 0,
        "sample_id": sample_id,
        "trajectory_group_id": group_id,
        "trajectory_id": f"trajectory-{group_id}",
        "view_id": "view-a",
        "trajectory_frame_index": 0,
        "split": split,
        "camera": camera,
        "projection": projection,
        "target_court": _target_court(),
        "metadata": {"fixture": True},
    }
    (sample_root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    record = {
        "sample_index": 0,
        "sample_id": sample_id,
        "trajectory_group_id": group_id,
        "trajectory_id": f"trajectory-{group_id}",
        "view_id": "view-a",
        "trajectory_frame_index": 0,
        "split": split,
        "shard_id": "shard-a",
        "width": _WIDTH,
        "height": _HEIGHT,
        "camera": camera,
        "projection": projection,
        "target_court": _target_court(),
        "directory": relative.as_posix(),
        "rgb": (relative / "rgb.npy").as_posix(),
        "rgb_preview": (relative / "rgb.png").as_posix(),
        "alpha": (relative / "alpha.npy").as_posix(),
        "alpha_preview": (relative / "alpha.png").as_posix(),
        "depth": (relative / "depth.npy").as_posix(),
        "depth_coordinate_space": "metric_scene_metres",
        "labels": (relative / "labels.json").as_posix(),
        "metadata": {"fixture": True},
    }
    manifest_path = dataset_root / "dataset.json"
    existing: list[dict[str, object]] = []
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
    existing.append(record)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "canonical_court_dataset_v3",
                "status": "completed",
                "scene_id": scene_id,
                "profile": "v3-fixture",
                "seed": 1,
                "sampling_policy": {},
                "metadata_fields": [],
                "trajectory_groups": [],
                "samples": existing,
                "rejected_samples": [],
                "metrics": {},
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )


def _synthetic_domain(workspace: Path) -> BenchmarkDomainSettings:
    return BenchmarkDomainSettings(
        domain="synthetic_test",
        display_name="synthetic_test",
        split="test",
        source=SyntheticCourtSourceConfig(
            kind="synthetic_court",
            schema="v3",
            court_scope="target_court",
            workspace_root=workspace,
            scene_ids=("B00", "B01"),
        ),
    )


def test_synthetic_ground_truth_uses_the_published_physical_order(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "scenes"
    _write_synthetic_scene(
        workspace,
        scene_id="B00",
        physical_order=tuple(range(14)),
        coverage_mode="near_full",
        sample_id="court-sample-000001",
        split="test",
        group_id="group-a",
    )
    # The input layer requires every split to be non-empty; the benchmark itself
    # only ever reads the test split.
    for split in ("train", "validation"):
        _write_synthetic_scene(
            workspace,
            scene_id="B00",
            physical_order=tuple(range(14)),
            coverage_mode="near_full",
            sample_id=f"court-sample-{split}",
            split=split,
            group_id=f"group-{split}",
        )
    _write_synthetic_scene(
        workspace,
        scene_id="B01",
        physical_order=CAMERA_VIEW_HALF_TURN_INDEX,
        coverage_mode="partial",
        sample_id="court-sample-000002",
        split="test",
        group_id="group-b",
    )
    domain = build_domain_records(
        _synthetic_domain(workspace), derived_target_root=tmp_path / "derived"
    )

    assert sorted(domain.records) == [
        "B00:court-sample-000001",
        "B01:court-sample-000002",
    ]
    identity = load_sample(
        domain, sample_ref(domain, domain.records["B00:court-sample-000001"])
    )
    half_turn = load_sample(
        domain, sample_ref(domain, domain.records["B01:court-sample-000002"])
    )
    template = court_template_xy(14).astype(np.float64)

    # Scene B00 publishes the identity permutation, B01 the half-turn one; the
    # template must follow the published order rather than assuming either.
    np.testing.assert_allclose(identity.template_xy, template)
    np.testing.assert_allclose(
        half_turn.template_xy, template[list(CAMERA_VIEW_HALF_TURN_INDEX)]
    )
    assert bool(identity.gt_visible.all())
    assert identity.strata == {
        "scene": "B00",
        "coverage_mode": "near_full",
        "visible_kp_count": "14",
    }
    assert half_turn.strata["coverage_mode"] == "partial"
    assert half_turn.ref.trajectory_group_id == "group-b"
    assert half_turn.image_rgb.shape == (_HEIGHT, _WIDTH, 3)


def _write_real_domain(root: Path) -> None:
    images = root / "court" / "images"
    images.mkdir(parents=True, exist_ok=True)
    points = [[float(6 + index * 2), float(6 + index)] for index in range(14)]
    records = {
        "train": [{"id": "train-1", "kps": points, "metric": 0.5}],
        "val": [
            {"id": "val-1", "kps": points, "metric": 0.25},
            {"id": "val-2", "kps": points, "metric": 0.75},
        ],
    }
    for split, entries in records.items():
        (root / "court" / f"data_{split}.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )
    for sample_id in ("train-1", "val-1", "val-2"):
        Image.new("RGB", (_WIDTH, _HEIGHT)).save(images / f"{sample_id}.png")


def test_real_validation_domain_labels_itself_as_validation(tmp_path: Path) -> None:
    _write_real_domain(tmp_path)
    settings = BenchmarkDomainSettings(
        domain="real_validation",
        display_name="real_validation",
        split="val",
        source=TennisCourtDetectorSourceConfig(
            kind="tennis_court_detector",
            root=tmp_path / "court",
            split_mapping={"train": "train", "val": "val", "test": None},
            excluded_sample_ids=(),
        ),
    )
    domain = build_domain_records(settings, derived_target_root=tmp_path / "derived")

    assert sorted(domain.records) == ["val-1", "val-2"]
    loaded = load_sample(domain, sample_ref(domain, domain.records["val-1"]))

    assert loaded.ref.domain == "real_validation"
    assert loaded.ref.split == "val"
    assert loaded.ref.trajectory_group_id is None
    assert loaded.strata["scene"] == "tennis_court_detector"
    assert loaded.strata["visible_kp_count"] == "14"
    assert "coverage_mode" not in loaded.strata


def test_real_validation_domain_never_reads_the_train_split(tmp_path: Path) -> None:
    """A validation-only benchmark must not preflight the production train split.

    The source keeps the production ``train -> train`` mapping, so the only
    thing that can keep the run off 6,630 unrelated train images is the
    explicit partial read.  Train is therefore deleted outright here: any eager
    preflight of it would fail the construction.
    """
    _write_real_domain(tmp_path)
    (tmp_path / "court" / "data_train.json").unlink()
    (tmp_path / "court" / "images" / "train-1.png").write_bytes(b"not a png")
    settings = BenchmarkDomainSettings(
        domain="real_validation",
        display_name="real_validation",
        split="val",
        source=TennisCourtDetectorSourceConfig(
            kind="tennis_court_detector",
            root=tmp_path / "court",
            split_mapping={"train": "train", "val": "val", "test": None},
            excluded_sample_ids=("train-1",),
        ),
    )

    domain = build_domain_records(settings, derived_target_root=tmp_path / "derived")

    assert sorted(domain.records) == ["val-1", "val-2"]
    assert domain.input_layer.available_splits == ("val",)
    input_layer = cast(TennisCourtDetectorInput, domain.input_layer)
    # The train quarantine is outside the evaluated split; it is reported as
    # unverified instead of being dropped without a trace.
    assert input_layer.deferred_excluded_sample_ids == frozenset({"train-1"})
