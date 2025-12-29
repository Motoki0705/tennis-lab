"""HRNet backbone + ConvGRU temporal head for ball detection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class ConvGRUCell(nn.Module):
    """A lightweight ConvGRU cell for spatial feature maps."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor, h_prev: Tensor | None = None) -> Tensor:
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0),
                self.hidden_channels,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )

        combined = torch.cat([x, h_prev], dim=1)
        zr = torch.sigmoid(self.conv_gates(combined))
        z, r = torch.split(zr, self.hidden_channels, dim=1)
        candidate = torch.tanh(
            self.conv_candidate(torch.cat([x, r * h_prev], dim=1))
        )
        h = (1 - z) * h_prev + z * candidate
        return h


class StackedConvGRU(nn.Module):
    """Multi-layer ConvGRU over a sequence of spatial feature maps.

    Args:
        input_channels: Number of channels of the input sequence features.
        hidden_dims: Hidden channel size for each ConvGRU layer (depth of the stack).
        kernel_size: Kernel size used in each ConvGRUCell.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dims: Sequence[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one element")

        self.hidden_dims = [int(h) for h in hidden_dims]
        self.layers = nn.ModuleList()

        in_ch = input_channels
        for hidden_ch in self.hidden_dims:
            self.layers.append(ConvGRUCell(in_ch, hidden_ch, kernel_size))
            in_ch = hidden_ch

    def forward(
        self,
        x_seq: Tensor,
        h_states: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Process a sequence of features with stacked ConvGRU layers.

        Args:
            x_seq: Input sequence of features, shape ``[B, T, C_in, H, W]``.
            h_states: Optional list of previous hidden states for each layer.
                Each element is a tensor of shape ``[B, C_l, H, W]``.

        Returns:
            y_seq: Output sequence from the top layer, shape ``[B, T, C_L, H, W]``.
            h_states: Final hidden states of all layers as a list of tensors.
        """
        if x_seq.dim() != 5:
            raise ValueError(
                f"Expected x_seq shape [B, T, C, H, W], got {tuple(x_seq.shape)}"
            )

        b, t, _, h, w = x_seq.shape
        num_layers = len(self.layers)

        if h_states is None:
            h_states = [None] * num_layers
        else:
            h_states = list(h_states)
            if len(h_states) != num_layers:
                raise ValueError(
                    f"h_states length {len(h_states)} does not match "
                    f"number of layers {num_layers}"
                )

        outputs: list[Tensor] = []
        for ti in range(t):
            x = x_seq[:, ti]  # [B, C_in, H, W]
            new_h: list[Tensor] = []
            for layer, h_prev in zip(self.layers, h_states):
                x = layer(x, h_prev)  # x becomes hidden for this layer
                new_h.append(x)
            h_states = new_h
            outputs.append(x)  # top-layer hidden

        y_seq = torch.stack(outputs, dim=1)  # [B, T, C_L, H, W]
        return y_seq, h_states


class TemporalConvGRUModel(nn.Module):
    """Sequence model using generic backbone features with a stacked ConvGRU head."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: int,
        frames_in: int,
        frames_out: int | None = None,
        stack_channels: bool = False,
        gru_hidden_channels: Sequence[int] | int | None = None,
        gru_kernel_size: int = 3,
        expects_sequence_input: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.feature_channels = int(feature_channels)

        self.frames_in = int(frames_in)
        self.frames_out = int(frames_out) if frames_out is not None else self.frames_in
        self.stack_channels = bool(stack_channels)
        self.expects_sequence_input = bool(expects_sequence_input)

        self._backbone_train_mode: bool | None = None
        self._backbone_frozen = False

        if gru_hidden_channels is None:
            hidden_dims = [self.feature_channels]
        elif isinstance(gru_hidden_channels, int):
            hidden_dims = [int(gru_hidden_channels)]
        else:
            hidden_dims = [int(h) for h in gru_hidden_channels]
        if not hidden_dims:
            raise ValueError("gru_hidden_channels must define at least one layer")

        kernel_size = int(gru_kernel_size)

        self.temporal_core = StackedConvGRU(
            input_channels=self.feature_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
        )
        self.head = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)

    def forward(
        self,
        frames: Tensor,
        h_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass with optional hidden state carry-over.

        Args:
            frames:
                Input video clip, shape ``[B, T, C, H, W]``.
            h_state:
                Optional list of hidden states for each ConvGRU layer.
                Each element has shape ``[B, C_l, H', W']`` where
                ``H', W'`` are feature map size.

        Returns:
            pred:
                Heatmaps for the last ``frames_out`` steps,
                shape ``[B, frames_out, H', W']``.
            h_state:
                Final hidden states list that can be fed into the next call.
        """
        if frames.dim() != 5:
            raise ValueError(
                f"Expected input shape [B, T, C, H, W], got {tuple(frames.shape)}"
            )
        b, t, c, h, w = frames.shape
        frames_flat = frames.view(b * t, c, h, w)

        # Backbone feature extraction
        feats_flat = self.backbone.forward_features(frames_flat)
        if feats_flat.dim() != 4:
            raise ValueError(
                f"Backbone features must be 4D tensor [B*T, C, H, W], got {tuple(feats_flat.shape)}"
            )
        feat_h, feat_w = feats_flat.shape[-2:]
        features = feats_flat.view(b, t, self.feature_channels, feat_h, feat_w)

        # Stacked ConvGRU over time
        seq_out, h_state = self.temporal_core(features, h_state)
        # seq_out: [B, T, C_L, feat_h, feat_w]

        # Use the last frames_out steps for prediction
        seq_out = seq_out[:, -self.frames_out :, :, :, :]  # [B, frames_out, C_L, H', W']
        b_out, t_out, c_out, fh, fw = seq_out.shape
        logits = self.head(seq_out.view(b_out * t_out, c_out, fh, fw))
        pred = logits.view(b_out, t_out, 1, fh, fw).squeeze(2)  # [B, frames_out, H', W']
        return pred, h_state

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: torch.device | str | None = "cpu",
    ) -> None:
        """Load pretrained weights into the HRNet backbone from a checkpoint.

        Accepts Lightning checkpoints (with ``state_dict`` entries) or raw state
        dicts. Only matching HRNet keys are loaded.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint must contain a state_dict: {checkpoint_path}")
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")

        backbone_state = self.backbone.state_dict()
        filtered_state: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            trimmed_key = key.removeprefix("model.")
            if trimmed_key in backbone_state:
                filtered_state[trimmed_key] = value

        if not filtered_state:
            raise ValueError(
                f"No matching backbone keys were found in checkpoint: {checkpoint_path}"
            )

        backbone_state.update(filtered_state)
        self.backbone.load_state_dict(backbone_state, strict=False)
        msg = f"Loaded {len(filtered_state)} backbone parameters from {checkpoint_path}"
        logger.info(msg)

    def freeze_backbone(self) -> None:
        """Disable gradient updates for the HRNet backbone."""
        self._backbone_train_mode = self.backbone.training
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True
        msg = "Backbone frozen"
        logger.info(msg)

    def unfreeze_backbone(self) -> None:
        """Re-enable gradient updates for the HRNet backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        if self._backbone_train_mode is not None:
            if self._backbone_train_mode:
                self.backbone.train()
            self._backbone_train_mode = None
        self._backbone_frozen = False
        msg = "Backbone unfrozen"
        logger.info(msg)


__all__ = ["HRNetConvGRU", "ConvGRUCell", "StackedConvGRU"]
