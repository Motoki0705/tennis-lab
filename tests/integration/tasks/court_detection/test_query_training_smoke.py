"""CPU fake-DINO forward/loss/backward smoke for the complete query seam."""

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
    CourtQueryRawOutput,
    CourtQueryTrainingResult,
)
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.models.query_encoder.model import CourtQueryEncoderModel
from src.tasks.court_detection.models.query_encoder.profiling import (
    profile_query_model,
    validate_profile_record,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.models.loading import DINOv3BackboneAdapter

pytestmark = pytest.mark.integration

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


class _FakeDINO(nn.Module):
    embed_dim = 8
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList((nn.Identity(),))

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "x_norm_patchtokens": self.patch_embed(images).flatten(2).transpose(1, 2)
        }


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=("synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"),
                output_channels=14,
                channel_names=tuple(f"kp_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


def _config(family: str) -> object:
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
                f"model/decoder=query_{family}_tiny",
                "model.backbone.train_mode=full",
            ],
        )


def _batch() -> dict[str, object]:
    batch_size, height, width = 2, 17, 19
    pose = CourtPoseTarget(
        translation_m=torch.tensor([0.0, -20.0, 10.0]),
        rotation=torch.eye(3),
        log_focal=torch.log(torch.tensor(100.0)),
        intrinsics=torch.tensor(
            [[100.0, 0.0, 9.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]]
        ),
        semantic_to_physical=torch.arange(14),
    )
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
            name: value.unsqueeze(0).expand(batch_size, *value.shape)
            for name, value in pose.to_mapping().items()
        },
        "image_size": torch.tensor([[height, width]] * batch_size, dtype=torch.long),
    }


@pytest.mark.parametrize("family", ["linear", "progressive", "dpt"])
def test_all_query_decoder_families_complete_finite_cpu_training_step(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDINO()
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(fake),
    )
    pair = build_court_detection_pair(_config(family), target_bundle=_bundle())
    model = cast(CourtQueryEncoderModel, pair.model)
    adapter = cast(CourtQueryModelIOAdapter, pair.adapter)
    batch = _batch()

    call = adapter.prepare_training_batch(batch)
    output = cast(CourtQueryRawOutput, model(*call.model_call.model_args))
    result = adapter.training_result(output, call)
    result.loss.backward()

    assert output.pose.values.shape == (2, 10)
    assert output.dense_logits["kp"].shape == (2, 14, 17, 19)
    assert set(result.dense_losses) == {"kp"}
    assert set(result.pose_losses) == {
        "pose_translation",
        "pose_rotation",
        "pose_focal",
    }
    assert bool(torch.isfinite(result.loss))
    assert fake.patch_embed.weight.grad is not None
    assert any(
        parameter.grad is not None for parameter in model.task_encoder.parameters()
    )


def test_query_lightning_logs_and_persists_every_typed_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(_FakeDINO()),
    )
    module = CourtDetectionLightningModule(
        _config("linear"),
        target_bundle=_bundle(),
    )
    logged: list[str] = []
    monkeypatch.setattr(
        module,
        "log",
        lambda name, *args, **kwargs: logged.append(name),
    )
    batch = _batch()

    result = module._shared_step(batch, "train")
    assert isinstance(result, CourtQueryTrainingResult)
    payload = module.test_prediction_payload(batch, {"output": result.output})

    assert set(logged) == {
        "train/loss",
        "train/loss_kp",
        "train/loss_pose_translation",
        "train/loss_pose_rotation",
        "train/loss_pose_focal",
    }
    assert "query_checkpoint_state" in module.hparams
    assert payload["pose_translation_m"].shape == (2, 3)
    assert payload["pose_rotation"].shape == (2, 3, 3)
    assert payload["pose_focal_px"].shape == (2,)
    assert payload["kp_keypoints_normalized"].shape == (2, 14, 1, 2)


def test_fake_dino_complete_model_cpu_profile_is_explicitly_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tasks.court_detection.models.query_encoder.backbone.load_dinov3_backbone",
        lambda **_: DINOv3BackboneAdapter(_FakeDINO()),
    )
    pair = build_court_detection_pair(_config("linear"), target_bundle=_bundle())
    model = cast(CourtQueryEncoderModel, pair.model)
    adapter = cast(CourtQueryModelIOAdapter, pair.adapter)

    record = profile_query_model(
        model,
        adapter,
        torch.zeros(1, 3, 17, 19),
        family="linear",
        size="tiny",
        warmup=0,
        repeats=1,
    )

    validate_profile_record(record, require_gpu_evidence=False)
    assert record["evidence"]["kind"] == "cpu_diagnostic"
    assert record["evidence"]["latency_is_adoption_evidence"] is False
    assert record["peak_memory"]["bytes"] is None
