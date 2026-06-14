"""Teacher/student DINOv3 network for LoRA self-distillation.

The student backbone carries trainable LoRA adapters; the teacher is an
exponential moving average (EMA) of the student and produces the distillation
targets. The pretrained DINOv3 weights stay frozen in both branches, so only the
LoRA adapters and the projection heads are learned. This keeps the backbone's
general visual capability intact while it adapts to the tennis domain.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from src.tasks.dino_ssl._dinov3 import load_dinov3_losses_and_head
from src.tasks.dino_ssl.models.backbone import apply_lora, build_dinov3_vit


class DinoSSLNetwork(nn.Module):
    """Student (LoRA, trainable) and teacher (EMA) DINOv3 branches with heads."""

    def __init__(self, model_cfg: Any) -> None:
        super().__init__()
        dino_head_cls, _, _, _ = load_dinov3_losses_and_head()

        self.ibot_enabled = bool(model_cfg.ibot.enabled)

        student_backbone = build_dinov3_vit(
            backbone_name=str(model_cfg.backbone_name),
            checkpoint_path=model_cfg.get("checkpoint_path"),
            load_pretrained=bool(model_cfg.get("load_pretrained", True)),
        )
        self.embed_dim = int(student_backbone.embed_dim)
        student_backbone = apply_lora(student_backbone, model_cfg.lora)

        # Teacher shares the architecture; it is never optimised directly.
        teacher_backbone = copy.deepcopy(student_backbone)
        for param in teacher_backbone.parameters():
            param.requires_grad_(False)

        dino_out_dim = int(model_cfg.dino.out_dim)
        head_kwargs = dict(
            hidden_dim=int(model_cfg.head.hidden_dim),
            bottleneck_dim=int(model_cfg.head.bottleneck_dim),
            nlayers=int(model_cfg.head.nlayers),
        )
        student_dino_head = dino_head_cls(self.embed_dim, dino_out_dim, **head_kwargs)
        student_dino_head.init_weights()
        teacher_dino_head = copy.deepcopy(student_dino_head)
        for param in teacher_dino_head.parameters():
            param.requires_grad_(False)

        modules: dict[str, nn.Module] = {
            "backbone": student_backbone,
            "dino_head": student_dino_head,
        }
        teacher_modules: dict[str, nn.Module] = {
            "backbone": teacher_backbone,
            "dino_head": teacher_dino_head,
        }

        if self.ibot_enabled:
            ibot_out_dim = int(model_cfg.ibot.out_dim)
            student_ibot_head = dino_head_cls(
                self.embed_dim, ibot_out_dim, **head_kwargs
            )
            student_ibot_head.init_weights()
            teacher_ibot_head = copy.deepcopy(student_ibot_head)
            for param in teacher_ibot_head.parameters():
                param.requires_grad_(False)
            modules["ibot_head"] = student_ibot_head
            teacher_modules["ibot_head"] = teacher_ibot_head

        self.student = nn.ModuleDict(modules)
        self.teacher = nn.ModuleDict(teacher_modules)

        # Initialise the teacher to exactly match the student.
        self._sync_teacher_from_student()

    @torch.no_grad()
    def _sync_teacher_from_student(self) -> None:
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters(), strict=True
        ):
            teacher_param.data.copy_(student_param.data)
        for teacher_buffer, student_buffer in zip(
            self.teacher.buffers(), self.student.buffers(), strict=True
        ):
            teacher_buffer.data.copy_(student_buffer.data)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA update: ``teacher = m * teacher + (1 - m) * student``."""
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters(), strict=True
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1.0 - momentum
            )

    # ---- student forward ----
    def forward_student(
        self,
        global_crops: list[torch.Tensor],
        local_crops: list[torch.Tensor],
        masks: list[torch.Tensor] | None,
    ) -> dict[str, Any]:
        backbone = self.student["backbone"]
        cls_features: list[torch.Tensor] = []
        global_cls_features: list[torch.Tensor] = []
        student_patch_logits: list[torch.Tensor] = []

        for index, crop in enumerate(global_crops):
            crop_masks = masks[index] if masks is not None else None
            out = backbone.forward_features(crop, masks=crop_masks)
            global_cls_features.append(out["x_norm_clstoken"])
            cls_features.append(out["x_norm_clstoken"])
            if self.ibot_enabled:
                student_patch_logits.append(
                    self.student["ibot_head"](out["x_norm_patchtokens"])
                )

        for crop in local_crops:
            out = backbone.forward_features(crop)
            cls_features.append(out["x_norm_clstoken"])

        cls_logits = torch.stack(
            [self.student["dino_head"](feat) for feat in cls_features], dim=0
        )
        result: dict[str, Any] = {
            "cls_logits": cls_logits,
            "global_cls_features": torch.stack(global_cls_features, dim=0),
        }
        if self.ibot_enabled:
            result["patch_logits"] = torch.cat(student_patch_logits, dim=0)
        return result

    # ---- teacher forward ----
    @torch.no_grad()
    def forward_teacher(self, global_crops: list[torch.Tensor]) -> dict[str, Any]:
        backbone = self.teacher["backbone"]
        cls_features: list[torch.Tensor] = []
        patch_features: list[torch.Tensor] = []
        for crop in global_crops:
            out = backbone.forward_features(crop)
            cls_features.append(self.teacher["dino_head"](out["x_norm_clstoken"]))
            if self.ibot_enabled:
                patch_features.append(
                    self.teacher["ibot_head"](out["x_norm_patchtokens"])
                )
        result: dict[str, Any] = {"cls_logits": torch.stack(cls_features, dim=0)}
        if self.ibot_enabled:
            result["patch_logits"] = torch.cat(patch_features, dim=0)
        return result


__all__ = ["DinoSSLNetwork"]
