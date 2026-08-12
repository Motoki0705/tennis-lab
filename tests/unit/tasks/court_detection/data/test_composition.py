"""Contract tests for Court source/target composition."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import cast

import numpy as np
import torch
from PIL import Image

from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    COURT_DATASET_SCHEMA,
    COURT_SAMPLE_SCHEMA,
)
from src.tasks.court_detection.configuration import (
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtRawSample,
    CourtSampleMetadata,
    CourtSampleRecord,
    CourtTargetKind,
    CourtTargetSpec,
    CourtTransformedSample,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.data.processing.geometry import CourtProcessingGeometry
from src.tasks.court_detection.data.processing.pipeline import (
    CourtProcessingPipeline,
)
from src.tasks.court_detection.data.processing.targets import CourtTargetBuilder
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.utils.data.heatmaps import generate_gaussian_heatmaps


def test_heatmap_default_preserves_one_map_per_point_and_max_reduces() -> None:
    centers = torch.tensor([[0.25, 0.25], [0.75, 0.75]])

    default = generate_gaussian_heatmaps((17, 19), centers, 0.02)
    explicit = generate_gaussian_heatmaps(
        (17, 19),
        centers,
        0.02,
        point_reduction="none",
    )
    reduced = generate_gaussian_heatmaps(
        (17, 19),
        centers.unsqueeze(0),
        0.02,
        visibility=torch.tensor([[True, True]]),
        point_reduction="max",
    )

    torch.testing.assert_close(default, explicit)
    assert default.shape == (2, 17, 19)
    assert reduced.shape == (1, 17, 19)
    torch.testing.assert_close(reduced[0], default.amax(dim=0))


def test_tennis_court_detector_input_emits_ordered_14_by_1_channels(tmp_path) -> None:
    root = tmp_path / "tcd"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(root / "images" / "sample.png")
    keypoints = [[float(index + 1), float(index + 2)] for index in range(14)]
    payload = [{"id": "sample", "kps": keypoints}]
    (root / "data_train.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "data_val.json").write_text(json.dumps(payload), encoding="utf-8")
    input_layer = TennisCourtDetectorInput(
        TennisCourtDetectorSourceConfig(
            kind="tennis_court_detector",
            root=root,
            split_mapping=MappingProxyType(
                {"train": "train", "val": "val", "test": "val"}
            ),
        ),
        target_store=CourtDerivedTargetStore(tmp_path / "derived"),
    )

    sample = input_layer.load(input_layer.records("train")[0])

    assert sample.keypoint_channels is not None
    assert sample.keypoint_channels.points_xy.shape == (14, 1, 2)
    assert sample.keypoint_channels.point_visible.shape == (14, 1)
    assert sample.court_instances[0].physical_indices.tolist() == list(range(14))


def _projection() -> dict[str, object]:
    courts: list[dict[str, object]] = []
    for court_index in range(2):
        classes: list[dict[str, object]] = []
        for class_id, class_name in enumerate(SEMANTIC_CLASS_NAMES):
            points: list[dict[str, object]] = []
            for point_index, physical_index in enumerate(
                PHYSICAL_INDICES_BY_CLASS[class_id]
            ):
                renderer_visible = not (
                    court_index == 0 and class_id == 0 and point_index == 0
                )
                points.append(
                    {
                        "physical_index": physical_index,
                        "uv": [
                            float((physical_index + court_index) % 8),
                            float((physical_index + court_index) % 6),
                        ],
                        "camera_depth_m": 10.0,
                        "scene_xyz_m": [0.0, 0.0, 0.0],
                        "in_front": True,
                        "in_frame": True,
                        "renderer_visible": renderer_visible,
                    }
                )
            classes.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "renderer_visible": any(
                        point["renderer_visible"] for point in points
                    ),
                    "points": points,
                }
            )
        courts.append(
            {
                "court_instance_id": f"court-{court_index}",
                "coverage_mode": "full",
                "classes": classes,
            }
        )
    return {
        "camera_id": "camera-0",
        "resolution": [8, 6],
        "coverage_modes": ["full"],
        "visible_class_names": list(SEMANTIC_CLASS_NAMES),
        "visible_point_count": 27,
        "courts": courts,
    }


def test_synthetic_input_consumes_manifest_paths_and_renderer_visibility(
    tmp_path,
) -> None:
    root = tmp_path / "B00" / "datasets" / "court"
    sample_dir = root / "samples" / "sample-0"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "rgb.npy", np.zeros((6, 8, 3), dtype=np.float32))
    projection = _projection()
    camera = {"camera_id": "camera-0"}
    metadata = {"source": "test"}
    labels = {
        "schema": COURT_SAMPLE_SCHEMA,
        "sample_index": 0,
        "sample_id": "sample-0",
        "trajectory_group_id": "group-0",
        "trajectory_id": "trajectory-0",
        "view_id": "view-0",
        "trajectory_frame_index": 0,
        "split": "train",
        "camera": camera,
        "projection": projection,
        "metadata": metadata,
    }
    (sample_dir / "labels.json").write_text(
        json.dumps(labels),
        encoding="utf-8",
    )
    record = {
        "sample_index": 0,
        "sample_id": "sample-0",
        "trajectory_group_id": "group-0",
        "trajectory_id": "trajectory-0",
        "view_id": "view-0",
        "trajectory_frame_index": 0,
        "split": "train",
        "shard_id": "shard-0",
        "width": 8,
        "height": 6,
        "camera": camera,
        "projection": projection,
        "directory": "samples/sample-0",
        "rgb": "samples/sample-0/rgb.npy",
        "rgb_preview": "samples/sample-0/rgb.png",
        "alpha": "samples/sample-0/alpha.npy",
        "alpha_preview": "samples/sample-0/alpha.png",
        "depth": "samples/sample-0/depth.npy",
        "depth_coordinate_space": "camera",
        "labels": "samples/sample-0/labels.json",
        "metadata": metadata,
    }
    manifest = {
        "schema": COURT_DATASET_SCHEMA,
        "status": "completed",
        "scene_id": "B00",
        "profile": "test",
        "seed": 1,
        "sampling_policy": {},
        "metadata_fields": [],
        "trajectory_groups": [],
        "samples": [record],
        "rejected_samples": [],
        "metrics": {},
        "diagnostics": {},
    }
    (root / "dataset.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    input_layer = SyntheticCourtInput(
        SyntheticCourtSourceConfig(
            kind="synthetic_court",
            workspace_root=tmp_path,
            scene_ids=("B00",),
        ),
        target_store=CourtDerivedTargetStore(tmp_path / "derived"),
    )

    sample = input_layer.load(input_layer.records("train")[0])

    assert sample.sample_id == "B00:sample-0"
    assert sample.keypoint_channels is not None
    assert sample.keypoint_channels.points_xy.shape == (7, 4, 2)
    assert sample.keypoint_channels.point_visible.shape == (7, 4)
    assert not bool(sample.keypoint_channels.point_visible[0, 0])
    assert bool(sample.court_instances[0].point_visible[0])
    assert len(sample.court_instances) == 2


def test_processing_pipeline_samples_geometry_once_for_all_targets(tmp_path) -> None:
    record = CourtSampleRecord(
        sample_id="sample",
        split="train",
        image_path=tmp_path / "unused.png",
        annotation_path=tmp_path / "unused.json",
        derived_key="train/sample",
        dense_target_refs={},
        payload={},
    )
    metadata = CourtSampleMetadata(
        source_kind="tennis_court_detector",
        source_schema="test_source",
        source_sample_id="sample",
        scene_id=None,
        provenance={},
    )
    raw = CourtRawSample(
        sample_id="sample",
        image=Image.new("RGB", (8, 8)),
        keypoint_channels=None,
        court_instances=(),
        dense_target_refs={},
        metadata=metadata,
    )

    class _Input:
        def load(self, selected):
            assert selected is record
            return raw

    class _Geometry:
        def __init__(self):
            self.sample_calls = 0
            self.apply_calls = 0
            self.plan = object()

        def sample(self, selected):
            assert selected is raw
            self.sample_calls += 1
            return self.plan

        def apply(self, selected, *, dense_targets, plan):
            assert selected is raw
            assert plan is self.plan
            assert set(dense_targets) == set()
            self.apply_calls += 1
            return CourtTransformedSample(
                sample_id="sample",
                image_tensor=torch.zeros(3, 8, 8),
                image_size=torch.tensor([8, 8], dtype=torch.long),
                keypoint_channels=None,
                court_instances=(),
                dense_targets={},
                horizontal_flipped=False,
                metadata=metadata,
            )

    class _Builder:
        def __init__(self, kind):
            self.spec = CourtTargetSpec(
                kind=kind,
                schema=f"test_{kind}",
                output_channels=1,
                channel_names=(kind,),
                target_dtype=torch.float32,
                precomputed=False,
            )
            self.seen: list[int] = []

        def preflight(self, records):
            assert records == (record,)

        def load_dense(self, selected):
            assert selected is raw
            return {}

        def build(self, selected):
            self.seen.append(id(selected))
            return torch.tensor(1.0)

    geometry = _Geometry()
    first = _Builder("kp")
    second = _Builder("line")
    pipeline = CourtProcessingPipeline(
        input_layer=cast(CourtInput, _Input()),
        geometry=cast(CourtProcessingGeometry, geometry),
        target_builders=cast(
            "tuple[CourtTargetBuilder, ...]",
            (first, second),
        ),
    )
    pipeline.preflight((record,))

    result = pipeline.process(record)

    assert geometry.sample_calls == 1
    assert geometry.apply_calls == 1
    assert first.seen == second.seen
    targets = cast("dict[CourtTargetKind, object]", result["targets"])
    assert tuple(targets) == ("kp", "line")
