#!/usr/bin/env python3
"""Inspect the forward output structure of a loaded DINO Swin backbone."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from checkpoints.DINO.scripts.load_dino_swin_backbone import (
    BACKBONE_CHOICES,
    load_swin_backbone_state,
)


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
        default=DINO_WEIGHTS_DIR / "swin_backbone_state_checkpoint0027_5scale.pth",
        help="Path to a trimmed Swin backbone checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        default="swin_L_384_22k",
        choices=BACKBONE_CHOICES,
        help="Backbone architecture defined in tmp_DINO Swin wrapper.",
    )
    parser.add_argument(
        "--pretrain-img-size",
        type=int,
        default=384,
        help="Pretraining image size for the Swin backbone.",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="Enable dilation in the last Swin stage.",
    )
    parser.add_argument(
        "--return-interm-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Intermediate Swin stages returned by the wrapper.",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Enable Swin activation checkpointing during model construction.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="Dummy input height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the dummy forward pass.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading the trimmed backbone checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, load_result = load_swin_backbone_state(
        args.checkpoint,
        backbone=args.backbone,
        pretrain_img_size=args.pretrain_img_size,
        dilation=args.dilation,
        return_interm_indices=args.return_interm_indices,
        use_checkpoint=args.use_checkpoint,
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
