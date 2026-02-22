"""TrackNetV3 heatmap wrapper for ball_detection training."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from src.ball_detection.models.third_party_loader import load_tracknetv3_tracknet_class


class TrackNetV3HeatmapModel(nn.Module):
    """Wrap third_party TrackNetV3 TrackNet into the local model interface."""

    def __init__(
        self,
        *,
        seq_len: int,
        pretrained_checkpoint: str | Path | None = None,
        strict_pretrained_load: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        tracknet_cls = load_tracknetv3_tracknet_class()
        self.model = tracknet_cls(in_dim=self.seq_len * 3, out_dim=self.seq_len)

        if pretrained_checkpoint is not None:
            self.load_pretrained_checkpoint(
                checkpoint_path=pretrained_checkpoint,
                strict=bool(strict_pretrained_load),
            )

    def load_pretrained_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TrackNetV3 checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint must be a dict, got: {type(checkpoint)}")

        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state_dict, dict):
            raise TypeError(f"State dict must be a dict, got: {type(state_dict)}")

        target_state = self.model.state_dict()
        loadable: dict[str, Tensor] = {}
        for raw_key, tensor in state_dict.items():
            candidates = [
                raw_key,
                str(raw_key).removeprefix("model."),
                str(raw_key).removeprefix("module."),
                str(raw_key).removeprefix("model.").removeprefix("module."),
            ]
            for candidate in candidates:
                if candidate in target_state and target_state[candidate].shape == tensor.shape:
                    loadable[candidate] = tensor
                    break

        if not loadable:
            raise ValueError(f"No compatible TrackNetV3 weights found in: {ckpt_path}")

        merged = target_state
        merged.update(loadable)
        missing, unexpected = self.model.load_state_dict(merged, strict=False)

        if strict and (missing or unexpected):
            raise ValueError(
                "Strict checkpoint load failed with "
                f"missing={missing}, unexpected={unexpected}"
            )

    def forward(self, frames: Tensor, frame_mask: Tensor | None = None) -> dict[str, Tensor]:
        _ = frame_mask
        squeeze_time = False

        if frames.dim() == 5:
            batch_size, seq_len, channels, height, width = frames.shape
            if channels != 3:
                raise ValueError(f"TrackNetV3 expects RGB frames (C=3), got channels={channels}.")
            if seq_len != self.seq_len:
                raise ValueError(
                    f"TrackNetV3 expects seq_len={self.seq_len}, got seq_len={seq_len}."
                )
            net_input = frames.reshape(batch_size, seq_len * channels, height, width)
        elif frames.dim() == 4:
            batch_size, channels, height, width = frames.shape
            if channels == self.seq_len * 3:
                net_input = frames
                squeeze_time = True
            else:
                raise ValueError(
                    "For rank-4 input, TrackNetV3 expects stacked channels with "
                    f"C={self.seq_len * 3}, got C={channels}."
                )
        else:
            raise ValueError(
                "frames must have shape [B, T, 3, H, W] or [B, T*3, H, W], "
                f"got {tuple(frames.shape)}"
            )

        probs = self.model(net_input)
        eps = 1e-6
        logits = torch.logit(torch.clamp(probs, min=eps, max=1.0 - eps))

        if squeeze_time:
            logits = logits[:, 0]

        return {"heatmap_logits": logits}

    @classmethod
    def from_config(cls, config: dict | DictConfig | None) -> TrackNetV3HeatmapModel:
        cfg = config or {}
        model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
        return cls(
            seq_len=int(model_cfg.get("seq_len", 8)),
            pretrained_checkpoint=model_cfg.get("pretrained_checkpoint"),
            strict_pretrained_load=bool(model_cfg.get("strict_pretrained_load", False)),
        )
