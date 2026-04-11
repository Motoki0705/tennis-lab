#!/usr/bin/env python3
"""Inspect the forward output structure of a loaded DINO backbone.

This script loads a trimmed DINO backbone-body checkpoint via
`load_backbone_body_state`, runs a dummy tensor through the returned model, and
prints the output container type plus each feature map's key and shape.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from load_dino_backbone import load_backbone_body_state


DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get(
        "MODEL_WEIGHTS_ROOT",
        "/mnt/d/weights" if Path("/mnt/d/weights").exists() else "D:/weights",
    )
)
DINO_WEIGHTS_DIR = DEFAULT_WEIGHTS_ROOT / "DINO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DINO_WEIGHTS_DIR / "backbone_body_state.pth",
        help="Path to a trimmed backbone-body checkpoint.",
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
        "--height",
        type=int,
        default=800,
        help="Dummy input height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Dummy input width.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the dummy forward pass.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading the backbone-body checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, load_result = load_backbone_body_state(
        args.checkpoint,
        backbone=args.backbone,
        dilation=args.dilation,
        return_interm_indices=args.return_interm_indices,
        strict=args.strict,
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dummy = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    with torch.no_grad():
        outputs = model(dummy)

    print("Load result")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  missing_keys: {list(load_result.missing_keys)}")
    print(f"  unexpected_keys: {list(load_result.unexpected_keys)}")
    print("Dummy input")
    print(f"  shape: {tuple(dummy.shape)}")
    print(f"  dtype: {dummy.dtype}")
    print(f"  device: {dummy.device}")
    print("Forward output")
    print(f"  type: {type(outputs).__name__}")

    for key, value in outputs.items():
        print(
            f"  [{key}] shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        )


if __name__ == "__main__":
    main()
