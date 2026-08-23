"""Fake-DINO compose/forward/backward tests for the query raw-output seam."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.geometry.pose import CourtPoseTarget
from src.tasks.court_detection.model_io.adapters import CourtQueryModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtModelIOError,
    CourtQueryPrediction,
    CourtQueryRawOutput,
)
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.models.query_encoder.contracts import CourtPose10DRaw
from src.utils.models.loading import DINOv3BackboneAdapter

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


class FakePatchDINO(nn.Module):
    embed_dim = 8
    patch_size = 4

    def __init__(self, *, extra_token: bool = False) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList((nn.Identity(),))
        self.extra_token = extra_token
        self.seen_shape: tuple[int, ...] | None = None

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_shape = tuple(images.shape)
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        if self.extra_token:
            tokens = torch.cat((tokens.new_zeros(tokens.shape[0], 1, 8), tokens), dim=1)
        return {"x_norm_patchtokens": tokens}


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=(
                    "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
                ),
                output_channels=14,
                channel_names=tuple(f"kp_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


def _config() -> object:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "data/source=synthetic_court",
                "data.source.keypoint_court_scope=target_court",
                "data/processing=kp",
                "data/augmentation=pose_safe",
                "loss=query_pose",
                "model=query_encoder",
                "model.backbone.train_mode=full",
            ],
        )


def _training_batch(batch_size: int = 2) -> dict[str, object]:
    height, width = 16, 20
    pose = CourtPoseTarget(
        translation_m=torch.tensor([0.0, -20.0, 10.0]),
        rotation=torch.eye(3),
        log_focal=torch.log(torch.tensor(100.0)),
        intrinsics=torch.tensor(
            [[100.0, 0.0, 9.5], [0.0, 100.0, 7.5], [0.0, 0.0, 1.0]]
        ),
        semantic_to_physical=torch.arange(14),
    )
    pose_mapping = pose.to_mapping()
    return {
        "image": torch.zeros(batch_size, 3, height, width),
        "targets": {
            "kp": {
                "heatmap": torch.zeros(batch_size, 14, height, width),
                "points_xy": torch.full((batch_size, 14, 1, 2), 0.5),
                "point_visible": torch.ones(batch_size, 14, 1, dtype=torch.bool),
                "physical_indices": torch.arange(14)
                .view(1, 14, 1)
                .expand(batch_size, 14, 1),
            }
        },
        "pose_target": {
            key: value.unsqueeze(0).expand(batch_size, *value.shape)
            for key, value in pose_mapping.items()
        },
        "image_size": torch.tensor([[height, width]] * batch_size, dtype=torch.long),
    }


def test_fake_dino_query_factory_preserves_padding_raw_order_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePatchDINO()
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(fake),
    )
    pair = build_court_detection_pair(_config(), target_bundle=_bundle())
    adapter = cast(CourtQueryModelIOAdapter, pair.adapter)
    images = torch.zeros(2, 3, 17, 19)

    prepared = adapter.prepare_images(images)
    raw = cast(CourtQueryRawOutput, pair.model(*prepared.model_args))
    adapter.validate_output(raw, call=prepared)
    loss = raw.pose.values.square().mean() + sum(
        logits.square().mean() for logits in raw.dense_logits.values()
    )
    loss.backward()

    assert fake.seen_shape == (2, 3, 20, 20)
    assert prepared.patch_batch.grid_hw == (5, 5)
    assert prepared.patch_batch.padding_hw == (3, 1)
    assert raw.pose.values.shape == (2, 10)
    assert raw.dense_logits["kp"].shape == (2, 14, 17, 19)
    assert fake.patch_embed.weight.grad is not None
    assert any(
        parameter.grad is not None
        for name, parameter in pair.model.named_parameters()
        if name.startswith("task_encoder")
    )


def test_fake_dino_special_token_fails_before_query_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePatchDINO(extra_token=True)
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(fake),
    )
    pair = build_court_detection_pair(_config(), target_bundle=_bundle())

    with pytest.raises(CourtModelIOError, match="token count"):
        pair.build_call({"image": torch.zeros(1, 3, 16, 20)})


def test_query_training_seam_requires_typed_dense_and_pose_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePatchDINO()
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(fake),
    )
    pair = build_court_detection_pair(_config(), target_bundle=_bundle())
    adapter = cast(CourtQueryModelIOAdapter, pair.adapter)

    with pytest.raises(CourtModelIOError, match="targets mapping"):
        adapter.prepare_training_batch({"image": torch.zeros(1, 3, 16, 20)})


def test_query_training_result_has_explicit_weighted_pose_and_dense_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePatchDINO()
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(fake),
    )
    pair = build_court_detection_pair(_config(), target_bundle=_bundle())
    adapter = cast(CourtQueryModelIOAdapter, pair.adapter)
    batch = _training_batch()

    call = adapter.prepare_training_batch(batch)
    output = cast(CourtQueryRawOutput, pair.model(*call.model_call.model_args))
    result = adapter.training_result(output, call)
    result.loss.backward()

    assert set(result.dense_losses) == {"kp"}
    assert set(result.pose_losses) == {
        "pose_translation",
        "pose_rotation",
        "pose_focal",
    }
    expected = result.dense_losses["kp"] + sum(result.pose_losses.values())
    torch.testing.assert_close(result.loss, expected)
    assert bool(torch.isfinite(result.loss))
    assert fake.patch_embed.weight.grad is not None


def test_query_kp_contract_rejects_multi_point_target_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(FakePatchDINO()),
    )
    pair = build_court_detection_pair(_config(), target_bundle=_bundle())
    adapter = cast(CourtQueryModelIOAdapter, pair.adapter)
    batch = _training_batch(batch_size=1)
    kp = cast(dict[str, torch.Tensor], cast(dict[str, object], batch["targets"])["kp"])
    kp["points_xy"] = kp["points_xy"].expand(1, 14, 2, 2)
    kp["point_visible"] = kp["point_visible"].expand(1, 14, 2)
    kp["physical_indices"] = kp["physical_indices"].expand(1, 14, 2)

    with pytest.raises(CourtModelIOError, match=r"\(B,C,P,2\)|singleton"):
        adapter.prepare_training_batch(batch)


def test_query_prediction_persistence_is_typed_and_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(FakePatchDINO()),
    )
    adapter = cast(
        CourtQueryModelIOAdapter,
        build_court_detection_pair(_config(), target_bundle=_bundle()).adapter,
    )
    logits = torch.full((1, 14, 8, 10), -10.0)
    logits[:, :, 2, 3] = 9.0
    logits[:, :, 6, 8] = 10.0
    raw_pose = torch.tensor(
        [[0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0]]
    )

    prediction = adapter.test_payload(
        {},
        CourtQueryRawOutput(
            pose=CourtPose10DRaw(raw_pose),
            dense_logits={"kp": logits},
        ),
    )

    assert isinstance(prediction, CourtQueryPrediction)
    dense = cast(dict[str, torch.Tensor], prediction.dense["kp"])
    assert dense["keypoints_normalized"].shape == (1, 14, 1, 2)
    assert dense["scores"].shape == (1, 14, 1)
    torch.testing.assert_close(
        dense["keypoints_normalized"][0, 0, 0],
        torch.tensor([8.0 / 9.0, 6.0 / 7.0]),
    )
    assert prediction.pose.translation_m.shape == (1, 3)
    assert prediction.pose.rotation.shape == (1, 3, 3)
