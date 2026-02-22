"""HRNet backbone with temporal ConvGRU head for heatmap prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from src.ball_detection.models.third_party_loader import load_wasb_hrnet_class
from src.wasb.models.ball_detection.temporal_conv_gru import StackedConvGRU


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_wasb_model_config_path() -> Path:
    return _repo_root() / "third_party" / "WASB-SBDT" / "src" / "configs" / "model" / "wasb.yaml"


class WASBHRNetBackboneAdapter(nn.Module):
    """Adapter to expose `forward_features` from third_party WASB HRNet."""

    def __init__(self, backbone_cfg: DictConfig) -> None:
        super().__init__()
        hrnet_cls = load_wasb_hrnet_class()
        self.backbone = hrnet_cls(backbone_cfg)

    @property
    def feature_channels(self) -> int:
        return int(self.backbone.final_layers[0].in_channels)

    @property
    def input_channels(self) -> int:
        return int(self.backbone._frames_in * 3)

    def forward_features(self, x: Tensor) -> Tensor:
        hrnet = self.backbone

        x = hrnet.conv1(x)
        x = hrnet.bn1(x)
        x = hrnet.relu(x)
        x = hrnet.conv2(x)
        x = hrnet.bn2(x)
        x = hrnet.relu(x)
        x = hrnet.layer1(x)

        x_list = []
        for branch_idx in range(hrnet.stage2_cfg["NUM_BRANCHES"]):
            if hrnet.transition1[branch_idx] is not None:
                x_list.append(hrnet.transition1[branch_idx](x))
            else:
                x_list.append(x)
        y_list = hrnet.stage2(x_list)

        x_list = []
        for branch_idx in range(hrnet.stage3_cfg["NUM_BRANCHES"]):
            if hrnet.transition2[branch_idx] is not None:
                x_list.append(hrnet.transition2[branch_idx](y_list[-1]))
            else:
                x_list.append(y_list[branch_idx])
        y_list = hrnet.stage3(x_list)

        x_list = []
        for branch_idx in range(hrnet.stage4_cfg["NUM_BRANCHES"]):
            if hrnet.transition3[branch_idx] is not None:
                x_list.append(hrnet.transition3[branch_idx](y_list[-1]))
            else:
                x_list.append(y_list[branch_idx])
        y_list = hrnet.stage4(x_list)

        feat = y_list[0]
        for deconv_idx in range(hrnet.num_deconvs):
            feat = hrnet.deconv_layers[deconv_idx][0](feat)
        return feat


class WASBHRNetTemporalModel(nn.Module):
    """Heatmap predictor combining WASB HRNet backbone and temporal ConvGRU."""

    def __init__(
        self,
        *,
        backbone_cfg: DictConfig,
        temporal_hidden_channels: Sequence[int],
        temporal_kernel_size: int,
        pretrained_backbone_checkpoint: str | Path | None = None,
        strict_pretrained_load: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = WASBHRNetBackboneAdapter(backbone_cfg)

        hidden_dims = [int(v) for v in temporal_hidden_channels]
        if not hidden_dims:
            raise ValueError("temporal_hidden_channels must contain at least one channel size.")

        self.temporal_core = StackedConvGRU(
            input_channels=self.backbone.feature_channels,
            hidden_dims=hidden_dims,
            kernel_size=int(temporal_kernel_size),
        )
        self.head = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)

        if pretrained_backbone_checkpoint is not None:
            self.load_backbone_checkpoint(
                pretrained_backbone_checkpoint,
                strict=bool(strict_pretrained_load),
            )

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint must be a dict, got: {type(checkpoint)}")

        state_dict: dict[str, Tensor] | None = None
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
        if state_dict is None:
            if all(isinstance(v, Tensor) for v in checkpoint.values()):
                state_dict = checkpoint  # type: ignore[assignment]
            else:
                raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        target_state = self.backbone.backbone.state_dict()
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
            raise ValueError(f"No compatible backbone weights found in: {ckpt_path}")

        merged = target_state
        merged.update(loadable)
        missing, unexpected = self.backbone.backbone.load_state_dict(merged, strict=False)

        if strict and (missing or unexpected):
            raise ValueError(
                "Strict checkpoint load failed with "
                f"missing={missing}, unexpected={unexpected}"
            )

    def forward(self, frames: Tensor, frame_mask: Tensor | None = None) -> dict[str, Tensor]:
        _ = frame_mask
        squeeze_time = False

        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
            squeeze_time = True
        if frames.dim() != 5:
            raise ValueError(
                "frames must have shape [B, T, C, H, W] or [B, C, H, W], "
                f"got {tuple(frames.shape)}"
            )

        batch_size, seq_len, channels, height, width = frames.shape
        expected_channels = self.backbone.input_channels
        if channels != expected_channels:
            raise ValueError(
                f"Expected channels={expected_channels} for WASB-HRNet input, got {channels}."
            )

        flat = frames.reshape(batch_size * seq_len, channels, height, width)
        feat_flat = self.backbone.forward_features(flat)
        feat_h, feat_w = feat_flat.shape[-2], feat_flat.shape[-1]

        feat_seq = feat_flat.view(
            batch_size,
            seq_len,
            self.backbone.feature_channels,
            feat_h,
            feat_w,
        )
        temporal_seq, _ = self.temporal_core(feat_seq)

        hidden_channels = temporal_seq.shape[2]
        logits = self.head(temporal_seq.view(batch_size * seq_len, hidden_channels, feat_h, feat_w))
        logits = logits.view(batch_size, seq_len, feat_h, feat_w)

        if squeeze_time:
            logits = logits[:, 0]

        return {"heatmap_logits": logits}

    @classmethod
    def from_config(cls, config: dict | DictConfig | None) -> WASBHRNetTemporalModel:
        cfg = config or {}
        model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}

        config_path = model_cfg.get("wasb_model_config_path")
        if config_path is None:
            config_path = _default_wasb_model_config_path()
        config_path = Path(str(config_path))
        if not config_path.is_absolute():
            config_path = _repo_root() / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"WASB model config was not found: {config_path}")

        backbone_cfg = OmegaConf.load(config_path)
        if not isinstance(backbone_cfg, DictConfig):
            backbone_cfg = OmegaConf.create(backbone_cfg)

        frames_in_override = model_cfg.get("backbone_frames_in")
        if frames_in_override is not None:
            backbone_cfg["frames_in"] = int(frames_in_override)
        frames_out_override = model_cfg.get("backbone_frames_out")
        if frames_out_override is not None:
            backbone_cfg["frames_out"] = int(frames_out_override)

        hidden_channels = model_cfg.get("temporal_hidden_channels", [64])
        if isinstance(hidden_channels, int):
            hidden_channels = [int(hidden_channels)]

        return cls(
            backbone_cfg=backbone_cfg,
            temporal_hidden_channels=[int(v) for v in hidden_channels],
            temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", 3)),
            pretrained_backbone_checkpoint=model_cfg.get("pretrained_backbone_checkpoint"),
            strict_pretrained_load=bool(model_cfg.get("strict_pretrained_load", False)),
        )
