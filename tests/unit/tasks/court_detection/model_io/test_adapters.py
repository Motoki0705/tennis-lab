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
    }
    return CourtTargetBundleSpec({kind: specs[kind] for kind in kinds})


def _loss_config(
    *,
    dense_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CourtLossConfig:
    kp_weight, seg_weight, line_weight = dense_weights
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
    bundle = _bundle("kp", "seg", "line")
    adapter = _adapter(bundle)
    model = _CountingCourtModel(bundle)
    adapter.validate_model_pair(model)
    call = adapter.prepare_training_batch(_batch(bundle))

    logits = model(*call.model_call.model_args)
    result = adapter.training_result(logits, call)
    assert isinstance(result, CourtTrainingResult)
    result.loss.backward()

    assert model.calls == 1
    assert set(result.logits) == {"kp", "seg", "line"}
    assert set(result.losses) == {"kp", "seg", "line"}
    assert model.bias.grad is not None


def test_dense_loss_result_exposes_raw_configured_effective_and_weighted_terms() -> None:
    bundle = _bundle("kp", "seg", "line")
    adapter = CourtModelIOAdapter(
        CourtModelSpec(target_bundle=bundle, in_channels=3, short_side=32),
        loss_config=_loss_config(dense_weights=(2.0, 3.0, 4.0)),
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
    bundle = _bundle("kp", "seg", "line")
    adapter = _adapter(bundle)
    kp_logits = torch.full((1, 2, 4, 5), -10.0)
    kp_logits[0, 0, 2, 3] = 10.0
    kp_logits[0, 1, 1, 1] = 10.0

    keypoints = adapter.decode_prediction(
        "kp",
        kp_logits,
        original_size_hw=(7, 9),
        subpixel_refine=False,
        max_peaks=1,
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
