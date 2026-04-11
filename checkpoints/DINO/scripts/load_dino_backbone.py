#!/usr/bin/env python3
"""Load only the DINO backbone from a full checkpoint.

This script mirrors the ResNet backbone path defined in
`third_party/DINO/models/dino/backbone.py` for the 5-scale config:

- `backbone='resnet50'`
- `dilation=False`
- `return_interm_indices=[0, 1, 2, 3]`

The full DINO checkpoint stores backbone parameters under `model` with keys like
`backbone.0.body.layer4.2.conv3.weight`. This script extracts those weights,
loads them into a DINO-style backbone wrapper, and can optionally save the
trimmed state dict for reuse.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get(
        "MODEL_WEIGHTS_ROOT",
        "/mnt/d/weights" if Path("/mnt/d/weights").exists() else "D:/weights",
    )
)
DINO_WEIGHTS_DIR = DEFAULT_WEIGHTS_ROOT / "DINO"


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen batch statistics and affine params.

    This matches `third_party/DINO/models/dino/backbone.py`.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class DINOBackbone(nn.Module):
    """ResNet backbone wrapper equivalent to DINO's `Backbone` path."""

    def __init__(
        self,
        name: str = "resnet50",
        dilation: bool = False,
        return_interm_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if return_interm_indices is None:
            return_interm_indices = [0, 1, 2, 3]
        if name not in {"resnet50", "resnet101"}:
            raise ValueError(f"Unsupported backbone: {name}")
        if return_interm_indices not in ([0, 1, 2, 3], [1, 2, 3], [3]):
            raise ValueError(
                "return_interm_indices must be one of [0,1,2,3], [1,2,3], [3]"
            )

        backbone = getattr(torchvision.models, name)(
            weights=None,
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=FrozenBatchNorm2d,
        )
        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            layer_name = f"layer{5 - len(return_interm_indices) + idx}"
            return_layers[layer_name] = str(layer_index)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.backbone_name = name
        self.return_interm_indices = return_interm_indices

    def forward(self, x: torch.Tensor):
        return self.body(x)


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)!r}")
    return checkpoint["model"] if "model" in checkpoint else checkpoint


def extract_backbone_body_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "backbone.0.body.",
) -> OrderedDict[str, torch.Tensor]:
    extracted = OrderedDict(
        (key[len(prefix) :], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    if not extracted:
        raise KeyError(f"No keys found with prefix {prefix!r}")
    return extracted


def summarize_state(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    keys = list(state_dict.keys())
    return {
        "count": len(keys),
        "sample_keys": keys[:10],
    }


def load_backbone_body_state(
    checkpoint_path: str | Path,
    *,
    backbone: str = "resnet50",
    dilation: bool = False,
    return_interm_indices: list[int] | None = None,
    strict: bool = True,
) -> tuple[DINOBackbone, nn.modules.module._IncompatibleKeys]:
    """Load a trimmed DINO backbone-body state dict into ``DINOBackbone``.

    This function is intended for checkpoints like
    ``D:/weights/DINO/backbone_body_state.pth``, where the saved
    keys correspond directly to the ResNet body, for example
    ``conv1.weight`` and ``layer4.2.conv3.weight``. The function constructs a
    ``DINOBackbone`` instance with the requested architecture settings, then
    loads the checkpoint into ``model.body``.

    Args:
        checkpoint_path: Path to a trimmed backbone checkpoint whose keys do not
            include the ``backbone.0.body.`` prefix.
        backbone: Backbone architecture name defined by the DINO ResNet
            backbone wrapper. Supported values are ``"resnet50"`` and
            ``"resnet101"``.
        dilation: Whether to replace the final ResNet stage stride with
            dilation.
        return_interm_indices: Intermediate feature levels returned by the DINO
            backbone wrapper. If ``None``, defaults to ``[0, 1, 2, 3]``.
        strict: Whether to enforce an exact key match in
            ``model.body.load_state_dict``.

    Returns:
        A tuple ``(model, load_result)`` where ``model`` is the constructed
        ``DINOBackbone`` instance and ``load_result`` is the
        ``load_state_dict`` compatibility result.
    """

    if return_interm_indices is None:
        return_interm_indices = [0, 1, 2, 3]

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DINOBackbone(
        name=backbone,
        dilation=dilation,
        return_interm_indices=return_interm_indices,
    )
    load_result = model.body.load_state_dict(state_dict, strict=strict)
    return model, load_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DINO_WEIGHTS_DIR / "checkpoint0011_5scale.pth",
        help="Path to the full DINO checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=["resnet50", "resnet101"],
        help="Backbone architecture defined in DINO backbone.py.",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="Enable dilation in the last ResNet stage.",
    )
    parser.add_argument(
        "--return-interm-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Intermediate layers returned by DINO Backbone.",
    )
    parser.add_argument(
        "--save-backbone-state",
        type=Path,
        help="Optional output path for the trimmed backbone state dict.",
    )
    parser.add_argument(
        "--save-full-backbone-module",
        type=Path,
        help="Optional output path for the loaded DINOBackbone module state dict.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading the extracted backbone weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_state = load_checkpoint_state(args.checkpoint)
    backbone_body_state = extract_backbone_body_state(full_state)

    print("Checkpoint analysis")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  total model keys: {len(full_state)}")
    print(f"  extracted backbone body keys: {len(backbone_body_state)}")
    print(f"  backbone: {args.backbone}")
    print(f"  dilation: {args.dilation}")
    print(f"  return_interm_indices: {args.return_interm_indices}")
    print("  sample extracted keys:")
    for key in list(backbone_body_state.keys())[:10]:
        print(f"    {key}")

    if args.save_backbone_state is not None:
        args.save_backbone_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone_body_state, args.save_backbone_state)
        print(f"Saved trimmed backbone state to: {args.save_backbone_state}")
        load_path = args.save_backbone_state
    else:
        load_path = args.checkpoint.with_suffix(".backbone_body.tmp.pth")
        torch.save(backbone_body_state, load_path)

    model, load_result = load_backbone_body_state(
        load_path,
        backbone=args.backbone,
        dilation=args.dilation,
        return_interm_indices=args.return_interm_indices,
        strict=args.strict,
    )

    print("Load result")
    print(f"  load_source: {load_path}")
    print(f"  missing_keys: {list(load_result.missing_keys)}")
    print(f"  unexpected_keys: {list(load_result.unexpected_keys)}")

    if args.save_full_backbone_module is not None:
        args.save_full_backbone_module.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_full_backbone_module)
        print(f"Saved loaded DINOBackbone state to: {args.save_full_backbone_module}")

    if args.save_backbone_state is None and load_path.exists():
        load_path.unlink()


if __name__ == "__main__":
    main()
