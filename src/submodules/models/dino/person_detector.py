"""IDEA-Research DINO frame-level person detector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.transforms import functional as transform_functional

from src.submodules.models._base import BaseInferenceModel
from src.utils.paths import PROJECT_ROOT

DEFAULT_DINO_CHECKPOINT = PROJECT_ROOT / "ckpt/dino/checkpoint0029_4scale_swin.pth"
COCO_PERSON_CLASS_ID = 1


@dataclass(frozen=True)
class PersonDetectionRequest:
    """One BGR uint8 frame to detect people in."""

    frame_bgr: NDArray[np.uint8]


@dataclass(frozen=True)
class PersonDetectionResult:
    """Person detections as xyxy pixel boxes and confidence scores."""

    boxes_xyxy: NDArray[np.float32]
    scores: NDArray[np.float32]


class DinoPersonDetector(
    BaseInferenceModel[PersonDetectionRequest, PersonDetectionResult]
):
    """Run the official four-scale Swin-L DINO COCO checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_DINO_CHECKPOINT,
        device: str | torch.device = "auto",
        confidence: float = 0.3,
        short_side: int = 800,
        max_long_side: int = 1333,
    ) -> None:
        super().__init__(device)
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if short_side <= 0 or max_long_side < short_side:
            raise ValueError(
                "Expected 0 < short_side <= max_long_side, got "
                f"{short_side} and {max_long_side}"
            )
        self.checkpoint = Path(checkpoint)
        self.confidence = confidence
        self.short_side = short_side
        self.max_long_side = max_long_side
        self._model: torch.nn.Module | None = None

    def _load_impl(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError(
                f"DINO multi-scale deformable attention requires CUDA, got {self.device}"
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"DINO checkpoint not found: {self.checkpoint}")
        try:
            from src.submodules.vendor.dino.models import build_dino
        except ModuleNotFoundError as error:
            if error.name == "MultiScaleDeformableAttention":
                raise RuntimeError(
                    "DINO CUDA extension is not installed. Run: "
                    "uv pip install -v --no-build-isolation "
                    "./src/submodules/vendor/dino/models/dino/ops"
                ) from error
            raise

        payload: Any = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(payload, Mapping)
            or "model" not in payload
            or "args" not in payload
        ):
            raise ValueError(
                "DINO checkpoint must contain both 'model' and 'args' entries"
            )
        _validate_checkpoint_args(payload["args"])
        args = _dino_4scale_swin_args(self.device)
        model, _, _ = build_dino(args)
        model.load_state_dict(payload["model"], strict=True)
        self._model = model.to(self.device).eval()

    def _unload_impl(self) -> None:
        self._model = None

    def _predict_impl(self, request: PersonDetectionRequest) -> PersonDetectionResult:
        if request.frame_bgr.ndim != 3 or request.frame_bgr.shape[2] != 3:
            raise ValueError(
                f"frame_bgr must have shape (H, W, 3), got {request.frame_bgr.shape}"
            )
        if request.frame_bgr.dtype != np.uint8:
            raise TypeError(
                f"frame_bgr must have dtype uint8, got {request.frame_bgr.dtype}"
            )
        assert self._model is not None
        height, width = request.frame_bgr.shape[:2]
        image = _preprocess_frame(
            request.frame_bgr,
            short_side=self.short_side,
            max_long_side=self.max_long_side,
        ).to(self.device)
        output = self._model(image.unsqueeze(0))
        return decode_person_detections(
            output,
            image_width=width,
            image_height=height,
            confidence=self.confidence,
        )


def _validate_checkpoint_args(args: Any) -> None:
    # Released checkpoints store CLI args but omit config-inherited fields.
    # The remaining architecture is checked by strict state-dict loading.
    expected = {"backbone": "swin_L_384_22k"}
    mismatches = {
        name: (expected_value, getattr(args, name, None))
        for name, expected_value in expected.items()
        if getattr(args, name, None) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{name}: expected {wanted!r}, got {actual!r}"
            for name, (wanted, actual) in mismatches.items()
        )
        raise ValueError(f"Unsupported DINO checkpoint architecture ({details})")


class _DinoConfig(SimpleNamespace):
    """Attribute config with the membership behavior expected upstream."""

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and hasattr(self, name)


def _dino_4scale_swin_args(device: torch.device) -> _DinoConfig:
    """Exact inference-relevant official DINO_4scale_swin.py configuration."""
    return _DinoConfig(
        device=str(device),
        num_classes=91,
        backbone="swin_L_384_22k",
        lr_backbone=1e-5,
        dilation=False,
        return_interm_indices=[1, 2, 3],
        backbone_freeze_keywords=None,
        use_checkpoint=False,
        position_embedding="sine",
        pe_temperatureH=20,
        pe_temperatureW=20,
        enc_layers=6,
        dec_layers=6,
        unic_layers=0,
        pre_norm=False,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.0,
        nheads=8,
        num_queries=900,
        query_dim=4,
        num_patterns=0,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        decoder_layer_noise=False,
        dln_xy_noise=0.2,
        dln_hw_noise=0.2,
        decoder_module_seq=["sa", "ca", "ffn"],
        decoder_sa_type="sa",
        dec_layer_number=None,
        transformer_activation="relu",
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        two_stage_type="standard",
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        dec_pred_bbox_embed_share=True,
        dec_pred_class_embed_share=True,
        use_detached_boxes_dec_out=False,
        use_dn=True,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=91,
        embed_init_tgt=True,
        masks=False,
        frozen_weights=None,
        aux_loss=True,
        matcher_type="HungarianMatcher",
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        cls_loss_coef=1.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        mask_loss_coef=1.0,
        dice_loss_coef=1.0,
        focal_alpha=0.25,
        no_interm_box_loss=False,
        interm_loss_coef=1.0,
        match_unstable_error=True,
        num_select=300,
        nms_iou_threshold=-1,
    )


def _preprocess_frame(
    frame_bgr: NDArray[np.uint8], *, short_side: int, max_long_side: int
) -> torch.Tensor:
    image = torch.from_numpy(np.ascontiguousarray(frame_bgr[..., ::-1])).permute(
        2, 0, 1
    )
    image = image.float().div_(255.0)
    height, width = image.shape[-2:]
    scale = short_side / min(height, width)
    if max(height, width) * scale > max_long_side:
        scale = max_long_side / max(height, width)
    target_size = [int(round(height * scale)), int(round(width * scale))]
    image = transform_functional.resize(image, target_size, antialias=True)
    return transform_functional.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def decode_person_detections(
    output: Mapping[str, torch.Tensor],
    *,
    image_width: int,
    image_height: int,
    confidence: float,
) -> PersonDetectionResult:
    """Decode COCO person logits and normalized cxcywh boxes."""
    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    if (
        logits.ndim != 3
        or boxes.ndim != 3
        or logits.shape[0] != 1
        or boxes.shape[0] != 1
    ):
        raise ValueError(
            "Expected batched DINO outputs with batch size 1, got "
            f"logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}"
        )
    scores = logits[0, :, COCO_PERSON_CLASS_ID].sigmoid()
    keep = scores >= confidence
    scores = scores[keep]
    boxes = boxes[0, keep]
    if scores.numel() == 0:
        return PersonDetectionResult(
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )
    order = scores.argsort(descending=True)
    scores = scores[order]
    boxes = boxes[order]
    center_x, center_y, box_width, box_height = boxes.unbind(-1)
    xyxy = torch.stack(
        [
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        ],
        dim=-1,
    ).clamp_(0.0, 1.0)
    scale = xyxy.new_tensor([image_width, image_height, image_width, image_height])
    xyxy = xyxy * scale
    return PersonDetectionResult(
        boxes_xyxy=xyxy.detach().cpu().numpy().astype(np.float32, copy=False),
        scores=scores.detach().cpu().numpy().astype(np.float32, copy=False),
    )
