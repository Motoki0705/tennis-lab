"""Contract tests for Court source/target composition."""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import torch
from PIL import Image

from src.tasks.court_detection.configuration import (
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.collate import court_detection_collate
from src.tasks.court_detection.data.contracts import (
    CourtInputCapability,
    CourtInputSpec,
    CourtKeypointChannels,
    CourtRawSample,
    CourtSampleMetadata,
    CourtSampleRecord,
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
    CourtTransformedSample,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.data.processing.geometry import CourtProcessingGeometry
from src.tasks.court_detection.data.processing.pipeline import (
    CourtProcessingPipeline,
)
from src.tasks.court_detection.data.processing.targets import (
    CourtTargetBuilder,
    KeypointTargetBuilder,
)
from src.utils.data.heatmaps import generate_gaussian_heatmaps
from tests.unit.tasks.court_detection.data.inputs.fixtures import write_tennis_records


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


def test_v3_target_court_point_capacity_and_float32_survive_collate() -> None:
    input_spec = CourtInputSpec(
        source_kind="synthetic_court",
        source_schema="canonical_court_dataset_v3",
        capabilities=frozenset({CourtInputCapability.KEYPOINT_CHANNELS}),
        keypoint_schema="synthetic_camera_view_kp14_v3_target_court",
        keypoint_channel_names=tuple(f"kp-{index}" for index in range(14)),
        keypoint_flip_permutation=tuple(range(14)),
    )
    builder = KeypointTargetBuilder(input_spec, sigma_ratio=0.02)
    points = torch.full((14, 1, 2), 2.0, dtype=torch.float64)
    channels = CourtKeypointChannels(
        channel_names=input_spec.keypoint_channel_names,
        points_xy=points,
        point_visible=torch.ones((14, 1), dtype=torch.bool),
        physical_indices=torch.arange(14).view(14, 1),
        horizontal_flip_permutation=tuple(range(14)),
    )
    sample = CourtTransformedSample(
        sample_id="target-court",
        image_tensor=torch.zeros(3, 32, 32),
        image_size=torch.tensor([32, 32], dtype=torch.long),
        keypoint_channels=channels,
        court_instances=(),
        dense_targets={},
        horizontal_flipped=False,
        metadata=CourtSampleMetadata(
            source_kind="synthetic_court",
            source_schema="canonical_court_dataset_v3",
            source_sample_id="target-court",
            scene_id="B00",
            provenance={},
        ),
    )

    target = cast(dict[str, torch.Tensor], builder.build(sample))
    batch = court_detection_collate(
        [
            {
                "image": sample.image_tensor,
                "targets": {"kp": target},
                "image_size": sample.image_size,
                "sample_id": sample.sample_id,
                "metadata": sample.metadata.to_dict(),
            }
        ],
        bundle=CourtTargetBundleSpec({"kp": builder.spec}),
    )
    collated = cast(
        dict[str, torch.Tensor], cast(dict[str, object], batch["targets"])["kp"]
    )

    assert builder.spec.schema == (
        "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
    )
    assert target["heatmap"].shape == (14, 32, 32)
    assert target["heatmap"].dtype == builder.spec.target_dtype == torch.float32
    assert target["points_xy"].dtype == builder.spec.target_dtype
    assert float(target["heatmap"][0, 2, 2]) > 0.99
    assert float(target["heatmap"][0, 25, 25]) < 1.0e-6
    assert collated["points_xy"].shape == (1, 14, 1, 2)
    assert collated["heatmap"].dtype == torch.float32
    assert collated["points_xy"].dtype == torch.float32
    assert collated["point_visible"].shape == (1, 14, 1)
    assert collated["physical_indices"].shape == (1, 14, 1)


def test_tennis_court_detector_input_emits_ordered_14_by_1_channels(tmp_path) -> None:
    root = tmp_path / "tcd"
    keypoints = [[float(index + 1), float(index + 2)] for index in range(14)]
    write_tennis_records(
        root,
        [
            {"id": "sample", "kps": keypoints, "split": "train"},
            {"id": "validation", "kps": keypoints, "split": "val"},
        ],
    )
    input_layer = TennisCourtDetectorInput(
        TennisCourtDetectorSourceConfig(
            kind="tennis_court_detector",
            root=root,
            split_mapping=MappingProxyType(
                {"train": "train", "val": "val", "test": None}
            ),
            excluded_sample_ids=(),
        ),
    )

    sample = input_layer.load(input_layer.records("train")[0])

    assert sample.keypoint_channels is not None
    assert sample.keypoint_channels.points_xy.shape == (14, 1, 2)
    assert sample.keypoint_channels.point_visible.shape == (14, 1)
    assert sample.court_instances[0].physical_indices.tolist() == list(range(14))


def test_processing_pipeline_samples_geometry_once_for_all_targets(
    tmp_path, monkeypatch
) -> None:
    record = CourtSampleRecord(
        sample_id="sample",
        split="train",
        image_path=tmp_path / "unused.png",
        annotation_path=tmp_path / "unused.json",
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
            from src.tasks.court_detection.data.processing.geometry import (
                CourtGeometryPlan,
            )

            self.plan = CourtGeometryPlan(torch.eye(3), (8, 8), False)

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

    monkeypatch.setattr(
        "src.tasks.court_detection.data.processing.pipeline.generate_online_targets",
        lambda *args, **kwargs: {},
    )
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
