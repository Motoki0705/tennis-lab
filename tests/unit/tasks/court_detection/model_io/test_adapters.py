"""Unit tests for the bundle-aware Court model-I/O boundary."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtModelIOError,
    CourtModelSpec,
    CourtSegmentationPrediction,
    CourtTrainingResult,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


def _bundle(*kinds: CourtTargetKind) -> CourtTargetBundleSpec:
    specs: dict[CourtTargetKind, CourtTargetSpec] = {
        "kp": CourtTargetSpec(
            kind="kp",
            schema="test_kp",
            output_channels=2,
            channel_names=("left", "right"),
            target_dtype=torch.float32,
            precomputed=False,
        ),
        "seg": CourtTargetSpec(
            kind="seg",
            schema="test_seg",
            output_channels=3,
            channel_names=("background", "a", "b"),
            target_dtype=torch.long,
            precomputed=True,
        ),
        "line": CourtTargetSpec(
            kind="line",
            schema="test_line",
            output_channels=1,
            channel_names=("line",),
            target_dtype=torch.float32,
            precomputed=True,
        ),
        "semantic_line": CourtTargetSpec(
            kind="semantic_line",
            schema="test_semantic_line",
            output_channels=4,
            channel_names=("background", "baseline", "sideline", "service"),
            target_dtype=torch.long,
            precomputed=True,
        ),
    }
    return CourtTargetBundleSpec({kind: specs[kind] for kind in kinds})


def _loss_config(
    *,
    dense_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> CourtLossConfig:
    kp_weight, seg_weight, line_weight, semantic_line_weight = dense_weights
    return CourtLossConfig.from_mapping(
        {
            "seg": {
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "weight": seg_weight,
            },
            "kp": {"focal_gamma": 2.0, "weight": kp_weight},
            "line": {
                "bce_weight": 1.0,
                "dice_weight": 1.0,
                "pos_weight": 1.0,
                "weight": line_weight,
            },
            "semantic_line": {
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "weight": semantic_line_weight,
            },
            "pose": {
                "enabled": False,
                "translation_weight": 0.0,
                "rotation_weight": 0.0,
                "focal_weight": 0.0,
            },
            "consistency": {
                "enabled": False,
                "weight": 0.0,
                "temperature": 1.0,
                "huber_delta": 0.01,
                "min_depth_m": 0.1,
                "depth_scale_m": 1.0,
                "cheirality_weight": 0.0,
                "warmup_fraction": 0.0,
                "gradient_flow": "both",
            },
        }
    )


class _CountingCourtModel(CourtHierarchicalModel):
    def __init__(self, bundle: CourtTargetBundleSpec) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.target_bundle_spec = bundle
        self.bias = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(
        self,
        images: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
        patch_valid_mask: torch.Tensor | None = None,
    ) -> dict[CourtTargetKind, torch.Tensor]:
        assert all(
            value is None
            for value in (
                feature_1,
                feature_2,
                feature_3,
                feature_4,
                patch_valid_mask,
            )
        )
        self.calls += 1
        return {
            kind: self.bias.expand(
                images.shape[0],
                spec.output_channels,
                images.shape[-2],
                images.shape[-1],
            )
            for kind, spec in self.target_bundle_spec.targets.items()
        }


def _adapter(bundle: CourtTargetBundleSpec) -> CourtModelIOAdapter:
    return CourtModelIOAdapter(
        CourtModelSpec(
            target_bundle=bundle,
            in_channels=3,
            short_side=32,
        ),
        loss_config=_loss_config(),
    )


def _batch(bundle: CourtTargetBundleSpec) -> dict[str, object]:
    targets: dict[str, object] = {}
    if "kp" in bundle.targets:
        targets["kp"] = {
            "heatmap": torch.zeros(1, 2, 8, 8),
            "points_xy": torch.zeros(1, 2, 1, 2),
            "point_visible": torch.ones(1, 2, 1, dtype=torch.bool),
            "physical_indices": torch.zeros(1, 2, 1, dtype=torch.long),
        }
    if "seg" in bundle.targets:
        targets["seg"] = torch.zeros(1, 8, 8, dtype=torch.long)
    if "line" in bundle.targets:
        targets["line"] = torch.zeros(1, 1, 8, 8)
    if "semantic_line" in bundle.targets:
        targets["semantic_line"] = torch.zeros(1, 8, 8, dtype=torch.long)
    return {
        "image": torch.zeros(1, 3, 8, 8),
        "targets": targets,
        "image_size": torch.tensor([[8, 8]], dtype=torch.long),
    }


def test_missing_head_target_fails_before_model_forward() -> None:
    bundle = _bundle("kp", "line")
    adapter = _adapter(bundle)
    model = _CountingCourtModel(bundle)
    adapter.validate_model_pair(model)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    del targets["line"]

    with pytest.raises(CourtModelIOError, match="exactly match"):
        adapter.prepare_training_batch(batch)

    assert model.calls == 0


def test_multi_head_training_runs_shared_model_once_and_backpropagates() -> None:
    bundle = _bundle("kp", "seg", "line", "semantic_line")
    adapter = _adapter(bundle)
    model = _CountingCourtModel(bundle)
    adapter.validate_model_pair(model)
    call = adapter.prepare_training_batch(_batch(bundle))

    logits = model(*call.model_call.model_args)
    result = adapter.training_result(logits, call)
    assert isinstance(result, CourtTrainingResult)
    result.loss.backward()

    assert model.calls == 1
    assert set(result.logits) == {"kp", "seg", "line", "semantic_line"}
    assert set(result.losses) == {"kp", "seg", "line", "semantic_line"}
    assert model.bias.grad is not None


def test_dense_loss_result_exposes_raw_configured_effective_and_weighted_terms() -> None:
    bundle = _bundle("kp", "seg", "line", "semantic_line")
    adapter = CourtModelIOAdapter(
        CourtModelSpec(target_bundle=bundle, in_channels=3, short_side=32),
        loss_config=_loss_config(dense_weights=(2.0, 3.0, 4.0, 5.0)),
    )
    call = adapter.prepare_training_batch(_batch(bundle))
    logits = {
        kind: torch.zeros(
            1,
            spec.output_channels,
            8,
            8,
            requires_grad=True,
        )
        for kind, spec in bundle.targets.items()
    }

    result = adapter.training_result(logits, call)
    assert isinstance(result, CourtTrainingResult)

    expected_weights: dict[CourtTargetKind, float] = {
        "kp": 2.0,
        "seg": 3.0,
        "line": 4.0,
        "semantic_line": 5.0,
    }
    assert result.losses is result.weighted_losses
    for kind, expected_weight in expected_weights.items():
        raw = result.raw_losses[kind]
        torch.testing.assert_close(
            result.configured_weights[kind],
            raw.new_tensor(expected_weight),
        )
        torch.testing.assert_close(
            result.effective_weights[kind],
            raw.new_tensor(expected_weight),
        )
        torch.testing.assert_close(
            result.weighted_losses[kind],
            raw * expected_weight,
        )
        torch.testing.assert_close(result.losses[kind], result.weighted_losses[kind])
    torch.testing.assert_close(
        result.raw_loss,
        torch.stack(tuple(result.raw_losses.values())).sum(),
    )
    torch.testing.assert_close(
        result.loss,
        torch.stack(tuple(result.weighted_losses.values())).sum(),
    )


def test_output_mapping_must_exactly_match_bundle() -> None:
    bundle = _bundle("kp", "line")
    adapter = _adapter(bundle)
    call = adapter.prepare_training_batch(_batch(bundle))

    with pytest.raises(CourtModelIOError, match="exactly match"):
        adapter.training_result(
            {"kp": torch.zeros(1, 2, 8, 8)},
            call,
        )


def test_decode_returns_typed_predictions_for_every_head() -> None:
    bundle = _bundle("kp", "seg", "line", "semantic_line")
    adapter = _adapter(bundle)
    kp_logits = torch.full((1, 2, 4, 5), -10.0)
    kp_logits[0, 0, 2, 3] = 10.0
    kp_logits[0, 1, 1, 1] = 10.0

    keypoints = adapter.decode_prediction(
        "kp",
        kp_logits,
        original_size_hw=(7, 9),
        subpixel_refine=False,
    )
    segmentation = adapter.decode_prediction(
        "seg",
        torch.zeros(1, 3, 4, 5),
        original_size_hw=(4, 5),
        subpixel_refine=False,
    )
    line = adapter.decode_prediction(
        "line",
        torch.zeros(1, 1, 4, 5),
        original_size_hw=(4, 5),
        subpixel_refine=False,
    )
    semantic_line = adapter.decode_prediction(
        "semantic_line",
        torch.zeros(1, 4, 4, 5),
        original_size_hw=(4, 5),
        subpixel_refine=False,
    )

    assert isinstance(keypoints, CourtKeypointPrediction)
    assert keypoints.keypoints.shape == (2, 1, 2)
    torch.testing.assert_close(
        keypoints.keypoints[:, 0],
        torch.tensor([[6.0, 4.0], [2.0, 2.0]]),
    )
    assert isinstance(segmentation, CourtSegmentationPrediction)
    assert segmentation.mask.shape == (4, 5)
    assert isinstance(line, CourtLinePrediction)
    torch.testing.assert_close(
        line.probability,
        torch.full((4, 5), 0.5),
    )
    assert isinstance(semantic_line, CourtSegmentationPrediction)
    assert semantic_line.mask.shape == (4, 5)


def test_decode_prediction_keeps_extra_peaks_only_when_requested() -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    logits = torch.full((1, 2, 9, 9), -20.0)
    logits[0, 0, 1, 1] = torch.logit(torch.tensor(0.9))
    logits[0, 0, 7, 7] = torch.logit(torch.tensor(0.7))
    logits[0, 1, 4, 4] = torch.logit(torch.tensor(0.8))
    default = adapter.decode_prediction(
        "kp",
        logits,
        original_size_hw=(9, 9),
        subpixel_refine=False,
    )
    multi = adapter.decode_prediction(
        "kp",
        logits,
        original_size_hw=(9, 9),
        subpixel_refine=False,
        max_peaks=2,
    )

    assert isinstance(default, CourtKeypointPrediction)
    assert default.keypoints.shape == (2, 1, 2)
    assert default.valid.tolist() == [[True], [True]]
    torch.testing.assert_close(
        default.keypoints[:, 0],
        torch.tensor([[1.0, 1.0], [4.0, 4.0]]),
    )
    assert multi.keypoints.shape == (2, 2, 2)
    assert multi.valid.tolist() == [[True, True], [True, False]]
    torch.testing.assert_close(
        multi.keypoints[0],
        torch.tensor([[1.0, 1.0], [7.0, 7.0]]),
    )


def _kp_head_payload(payload: dict[str, object]) -> dict[str, object]:
    predictions = payload["predictions"]
    assert isinstance(predictions, dict)
    kp_payload = predictions["kp"]
    assert isinstance(kp_payload, dict)
    return kp_payload


def test_test_payload_matches_the_supervised_point_capacity() -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    kp_logits = torch.full((1, 2, 8, 8), -20.0)
    kp_logits[0, 0, 2, 3] = torch.logit(torch.tensor(0.9))
    logits = {"kp": kp_logits}

    singleton = adapter.test_payload(batch, logits)
    singleton_kp = _kp_head_payload(singleton)
    singleton_keypoints = singleton_kp["keypoints_normalized"]
    singleton_scores = singleton_kp["scores"]
    singleton_valid = singleton_kp["valid"]
    assert isinstance(singleton_keypoints, torch.Tensor)
    assert isinstance(singleton_scores, torch.Tensor)
    assert isinstance(singleton_valid, torch.Tensor)
    assert singleton_keypoints.shape == (1, 2, 1, 2)
    assert singleton_scores.shape == (1, 2, 1)
    assert singleton_valid.tolist() == [[[True], [False]]]

    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    kp_target["points_xy"] = torch.zeros(1, 2, 2, 2)
    kp_target["point_visible"] = torch.ones(1, 2, 2, dtype=torch.bool)
    kp_target["physical_indices"] = torch.zeros(1, 2, 2, dtype=torch.long)

    multi = adapter.test_payload(batch, logits)
    multi_kp = _kp_head_payload(multi)
    multi_keypoints = multi_kp["keypoints_normalized"]
    multi_scores = multi_kp["scores"]
    assert isinstance(multi_keypoints, torch.Tensor)
    assert isinstance(multi_scores, torch.Tensor)
    assert multi_keypoints.shape == (1, 2, 2, 2)
    assert multi_scores.shape == (1, 2, 2)


def test_test_payload_requires_the_supervised_kp_target() -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    del batch["targets"]

    with pytest.raises(CourtModelIOError, match="targets mapping"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})


@pytest.mark.parametrize(
    "points_xy",
    [
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 1, 3),
        torch.zeros(2, 2, 1, 2),
    ],
)
def test_test_payload_rejects_malformed_point_targets(points_xy: torch.Tensor) -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    kp_target["points_xy"] = points_xy

    with pytest.raises(CourtModelIOError, match="points_xy"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})


def test_test_payload_rejects_empty_point_capacity() -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    kp_target["points_xy"] = torch.zeros(1, 2, 0, 2)

    with pytest.raises(CourtModelIOError, match="positive supervised point count"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})


def test_test_payload_rejects_missing_point_visibility() -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    del kp_target["point_visible"]

    with pytest.raises(CourtModelIOError, match="point_visible"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})


@pytest.mark.parametrize(
    "point_visible",
    [
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.ones(1, 2, 1),
    ],
)
def test_test_payload_rejects_malformed_point_visibility(
    point_visible: torch.Tensor,
) -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    kp_target["point_visible"] = point_visible

    with pytest.raises(CourtModelIOError, match="point_visible"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})


@pytest.mark.parametrize(
    "physical_indices",
    [
        torch.zeros(1, 2, 1).int(),
        torch.zeros(1, 2, 2, dtype=torch.long),
    ],
)
def test_test_payload_rejects_malformed_physical_indices(
    physical_indices: torch.Tensor,
) -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    kp_target["physical_indices"] = physical_indices

    with pytest.raises(CourtModelIOError, match="physical_indices"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})


def test_test_payload_rejects_non_finite_points() -> None:
    bundle = _bundle("kp")
    adapter = _adapter(bundle)
    batch = _batch(bundle)
    targets = batch["targets"]
    assert isinstance(targets, dict)
    kp_target = targets["kp"]
    assert isinstance(kp_target, dict)
    kp_target["points_xy"] = torch.full((1, 2, 1, 2), float("nan"))

    with pytest.raises(CourtModelIOError, match="finite"):
        adapter.test_payload(batch, {"kp": torch.zeros(1, 2, 8, 8)})
