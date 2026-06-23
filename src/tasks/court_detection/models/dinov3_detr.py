"""DINOv3-backed DETR decoder for semantic court segmentation.

The DINOv3 backbone code and pretrained weights are loaded from the vendored
``third_party/dinov3`` repository and remain subject to the DINOv3 License
Agreement in ``third_party/dinov3/LICENSE.md``.

The decoder follows the learned-query architecture described by DETR
(Carion et al., ECCV 2020), but is implemented here with PyTorch's
``TransformerDecoder`` rather than copied DETR source code. The segmentation
head follows the mask-classification formulation: each query predicts class
scores and a mask embedding, which is combined with dense pixel embeddings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.paths import resolve_project_path

if TYPE_CHECKING:
    from omegaconf import DictConfig


_DEFAULT_DINOV3_REPOSITORY = Path("third_party/dinov3")
_DEFAULT_DINOV3_CHECKPOINT = Path(
    "third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


class DINOv3BackboneAdapter(nn.Module):
    """Expose the patch-token subset of the dynamic DINOv3 hub API."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        dynamic_module = cast(Any, module)
        embed_dim = dynamic_module.embed_dim
        patch_size = dynamic_module.patch_size
        if not isinstance(embed_dim, int):
            raise TypeError("DINOv3 backbone embed_dim must be an integer.")
        if isinstance(patch_size, tuple):
            if len(patch_size) != 2 or len(set(patch_size)) != 1:
                raise ValueError("DINOv3DETR requires square backbone patches.")
            patch_size = patch_size[0]
        if not isinstance(patch_size, int):
            raise TypeError("DINOv3 backbone patch_size must be an integer.")

        self.module = module
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = cast(Any, self.module).forward_features(inputs)
        if not isinstance(outputs, Mapping):
            raise TypeError("DINOv3 forward_features must return a mapping.")
        patch_tokens = outputs.get("x_norm_patchtokens")
        if not isinstance(patch_tokens, torch.Tensor):
            raise TypeError(
                "DINOv3 forward_features did not return tensor patch tokens."
            )
        return {"x_norm_patchtokens": patch_tokens}


def load_dinov3_backbone(
    *,
    repository_path: str | Path = _DEFAULT_DINOV3_REPOSITORY,
    checkpoint_path: str | Path = _DEFAULT_DINOV3_CHECKPOINT,
    backbone_name: str = "dinov3_vitb16",
    strict: bool = True,
) -> DINOv3BackboneAdapter:
    """Load a DINOv3 backbone from the vendored repository and local weights."""
    repository = resolve_project_path(repository_path)
    checkpoint = resolve_project_path(checkpoint_path)
    if not repository.is_dir():
        raise FileNotFoundError(f"DINOv3 repository not found: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")

    backbone: nn.Module = torch.hub.load(
        str(repository),
        backbone_name,
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(
            "DINOv3 checkpoint must contain a state-dict mapping, "
            f"got {type(state).__name__}."
        )

    load_result = backbone.load_state_dict(state, strict=strict)
    if strict and (load_result.missing_keys or load_result.unexpected_keys):
        raise RuntimeError(
            "Unexpected DINOv3 checkpoint load result: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}."
        )
    return DINOv3BackboneAdapter(backbone)


class MLP(nn.Module):
    """Small feed-forward projection used for query mask embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")

        dimensions = [input_dim]
        dimensions.extend([hidden_dim] * (num_layers - 1))
        dimensions.append(output_dim)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(
                dimensions[:-1],
                dimensions[1:],
                strict=True,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for index, layer in enumerate(self.layers):
            output = layer(output)
            if index < len(self.layers) - 1:
                output = F.gelu(output)
        return output


class SinePositionEmbedding2D(nn.Module):
    """Generate a normalized 2D sine/cosine position embedding."""

    def __init__(self, hidden_dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError("Position embedding expects a 4D feature map.")

        batch_size, _, height, width = feature_map.shape
        num_axis_features = self.hidden_dim // 2
        dtype = feature_map.dtype
        device = feature_map.device

        y_coordinates = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=height,
            dtype=dtype,
            device=device,
        )
        x_coordinates = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=width,
            dtype=dtype,
            device=device,
        )
        dimension_indices = torch.arange(
            num_axis_features,
            dtype=dtype,
            device=device,
        )
        frequencies = self.temperature ** (
            2
            * torch.div(dimension_indices, 2, rounding_mode="floor")
            / num_axis_features
        )

        position_y = y_coordinates[:, None] / frequencies[None, :]
        position_x = x_coordinates[:, None] / frequencies[None, :]
        position_y = torch.stack(
            (position_y[:, 0::2].sin(), position_y[:, 1::2].cos()),
            dim=-1,
        ).flatten(1)
        position_x = torch.stack(
            (position_x[:, 0::2].sin(), position_x[:, 1::2].cos()),
            dim=-1,
        ).flatten(1)

        position = torch.cat(
            (
                position_y[:, None, :].expand(-1, width, -1),
                position_x[None, :, :].expand(height, -1, -1),
            ),
            dim=-1,
        )
        return position.reshape(1, height * width, self.hidden_dim).expand(
            batch_size, -1, -1
        )


class DETRSegmentationHead(nn.Module):
    """Predict query classes and masks, then compose semantic logits."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        mask_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, num_layers=3)

    def forward(
        self,
        query_features: torch.Tensor,
        pixel_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        class_logits = self.class_embed(query_features)
        mask_embeddings = self.mask_embed(query_features)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw",
            mask_embeddings,
            pixel_features,
        )
        return {"pred_logits": class_logits, "pred_masks": mask_logits}

    def semantic_logits(
        self,
        query_outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert mask-classification outputs to normalized dense logits."""
        class_probabilities = query_outputs["pred_logits"].softmax(dim=-1)[..., :-1]
        mask_probabilities = query_outputs["pred_masks"].sigmoid()
        semantic_scores = torch.einsum(
            "bqc,bqhw->bchw",
            class_probabilities,
            mask_probabilities,
        )
        semantic_probabilities = semantic_scores / semantic_scores.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        return semantic_probabilities.clamp_min(1e-6).log()


class DINOv3DETR(nn.Module):
    """Semantic segmentation model with DINOv3 and a plain DETR decoder."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 7,
        backbone_repository_path: str | Path = _DEFAULT_DINOV3_REPOSITORY,
        backbone_checkpoint_path: str | Path = _DEFAULT_DINOV3_CHECKPOINT,
        backbone_name: str = "dinov3_vitb16",
        backbone_strict: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_attention_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
        mask_dim: int = 256,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_attention_heads=num_attention_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            mask_dim=mask_dim,
        )

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone = load_dinov3_backbone(
            repository_path=backbone_repository_path,
            checkpoint_path=backbone_checkpoint_path,
            backbone_name=backbone_name,
            strict=backbone_strict,
        )
        backbone_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.memory_projection = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        self.position_embedding = SinePositionEmbedding2D(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=pre_norm,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.query_embeddings = nn.Embedding(num_queries, hidden_dim)

        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                hidden_dim,
                mask_dim,
                kernel_size=2,
                stride=2,
            ),
        )
        self.segmentation_head = DETRSegmentationHead(
            hidden_dim=hidden_dim,
            mask_dim=mask_dim,
            num_classes=self.num_classes,
        )

    @staticmethod
    def _validate_init_args(
        *,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        num_decoder_layers: int,
        num_attention_heads: int,
        feedforward_dim: int,
        dropout: float,
        mask_dim: int,
    ) -> None:
        if in_channels != 3:
            raise ValueError("DINOv3DETR requires 3-channel RGB input.")
        for name, value in (
            ("num_classes", num_classes),
            ("hidden_dim", hidden_dim),
            ("num_queries", num_queries),
            ("num_decoder_layers", num_decoder_layers),
            ("num_attention_heads", num_attention_heads),
            ("feedforward_dim", feedforward_dim),
            ("mask_dim", mask_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads.")
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @classmethod
    def from_config(cls, config: DictConfig) -> DINOv3DETR:
        """Create the model from a composed Hydra config."""
        model_cfg = config.get("model", {}) or {}
        backbone_cfg = model_cfg.get("backbone", {}) or {}
        decoder_cfg = model_cfg.get("decoder", {}) or {}
        head_cfg = model_cfg.get("segmentation_head", {}) or {}
        return cls(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 7)),
            backbone_repository_path=backbone_cfg.get(
                "repository_path",
                _DEFAULT_DINOV3_REPOSITORY,
            ),
            backbone_checkpoint_path=backbone_cfg.get(
                "checkpoint_path",
                _DEFAULT_DINOV3_CHECKPOINT,
            ),
            backbone_name=str(backbone_cfg.get("name", "dinov3_vitb16")),
            backbone_strict=bool(backbone_cfg.get("strict", True)),
            freeze_backbone=bool(backbone_cfg.get("freeze", True)),
            hidden_dim=int(decoder_cfg.get("hidden_dim", 256)),
            num_queries=int(decoder_cfg.get("num_queries", 100)),
            num_decoder_layers=int(decoder_cfg.get("num_layers", 6)),
            num_attention_heads=int(decoder_cfg.get("num_heads", 8)),
            feedforward_dim=int(decoder_cfg.get("feedforward_dim", 2048)),
            dropout=float(decoder_cfg.get("dropout", 0.1)),
            pre_norm=bool(decoder_cfg.get("pre_norm", False)),
            mask_dim=int(head_cfg.get("mask_dim", 256)),
        )

    def forward_query_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw DETR class and mask predictions for each query."""
        self._validate_forward_input(x)
        input_size = x.shape[-2:]
        patch_features = self._extract_patch_features(x)

        memory_map = self.memory_projection(patch_features)
        memory = memory_map.flatten(2).transpose(1, 2)
        memory = memory + self.position_embedding(memory_map)

        queries = self.query_embeddings.weight.unsqueeze(0).expand(
            x.shape[0],
            -1,
            -1,
        )
        query_features = self.decoder(tgt=queries, memory=memory)
        pixel_features = self.pixel_decoder(patch_features)
        query_outputs: dict[str, torch.Tensor] = self.segmentation_head(
            query_features,
            pixel_features,
        )
        query_outputs["pred_masks"] = F.interpolate(
            query_outputs["pred_masks"],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return query_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense semantic logits with shape ``(B, C, H, W)``."""
        return self.segmentation_head.semantic_logits(self.forward_query_outputs(x))

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_outputs = self.backbone.forward_features(x)
        else:
            backbone_outputs = self.backbone.forward_features(x)

        patch_tokens = backbone_outputs["x_norm_patchtokens"]
        patch_height = x.shape[-2] // self.patch_size
        patch_width = x.shape[-1] // self.patch_size
        expected_tokens = patch_height * patch_width
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "DINOv3 patch-token count does not match the input grid: "
                f"expected {expected_tokens}, got {patch_tokens.shape[1]}."
            )
        return patch_tokens.transpose(1, 2).reshape(
            x.shape[0],
            patch_tokens.shape[-1],
            patch_height,
            patch_width,
        )

    def _validate_forward_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"DINOv3DETR expects input with shape (B, C, H, W), got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )
        if min(x.shape[-2:]) < self.patch_size:
            raise ValueError(
                f"Input height and width must be at least {self.patch_size} pixels."
            )


__all__ = [
    "DETRSegmentationHead",
    "DINOv3BackboneAdapter",
    "DINOv3DETR",
    "SinePositionEmbedding2D",
    "load_dinov3_backbone",
]
