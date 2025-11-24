import torch
import torch.nn as nn


class DINOBackbone(nn.Module):
    """
    Thin wrapper around third_party/dinov3 backbones via torch.hub (source='local').

    - Default: use ViT-S/16 (pretrained checkpoint shipped in repo) to avoid network.
    - Exposes a forward_features_map(x) that returns a 2D feature map (B, C, H/16, W/16) for ViT.
    - Optionally, you can request multiple intermediate layers, but for ViT all share same resolution.
    """

    def __init__(
        self,
        backbone_name: str = "dinov3_vits16",
        weights_path: str | None = "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        vit_layers: tuple[int, ...] = (11,),
    ) -> None:
        super().__init__()
        # Load backbone locally
        # Note: for ViT, get_intermediate_layers(..., reshape=True) yields (B, C, H/patch, W/patch)
        self.model = torch.hub.load(
            "third_party/dinov3",
            backbone_name,
            source="local",
            weights=weights_path if weights_path is not None else True,
        )
        self.backbone_name = backbone_name
        self.vit_layers = vit_layers

    @property
    def embed_dim(self) -> int:
        # ViT exposes embed_dim; fall back to num_features if present
        if hasattr(self.model, "embed_dim"):
            return int(self.model.embed_dim)
        if hasattr(self.model, "num_features"):
            return int(self.model.num_features)
        raise AttributeError("Backbone missing embed dim attribute")

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a single 2D feature map (B, C, H', W').
        - For ViT-S/16, H' = H/16, W' = W/16
        """
        # Prefer using get_intermediate_layers with reshape=True to obtain (B,C,H,W)
        outs = self.model.get_intermediate_layers(
            x, n=self.vit_layers, reshape=True, return_class_token=False, norm=True
        )
        # outs is a tuple of tensors, take the last selected
        fmap = outs[-1]
        # Ensure channel-first
        assert fmap.dim() == 4, f"Expected BCHW, got shape {tuple(fmap.shape)}"
        return fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features_map(x)
