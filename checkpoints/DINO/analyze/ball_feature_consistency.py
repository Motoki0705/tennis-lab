#!/usr/bin/env python3
"""Analyze ball-point feature consistency across DINO backbone scales.

This script follows the official DINO evaluation preprocessing path from
`tmp_DINO/datasets/coco.py`:
- deterministic resize with short side 800 and max side 1333
- ImageNet mean/std normalization

For each frame in a clip, the script resizes the image as DINO evaluation does,
runs a DINO backbone on CUDA by default when available, samples the feature
vector at the resized ball coordinate for each returned scale, and summarizes
how consistent those features are over time.

Outputs:
- `ball_features.pt`: per-frame metadata and sampled feature tensors.
- `ball_feature_consistency_per_frame.csv`: per-frame norms and cosine values.
- `ball_feature_consistency_summary.csv`: scale-wise aggregate statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TVF

from checkpoints.DINO.scripts.load_dino_backbone import load_backbone_body_state
from checkpoints.DINO.scripts.load_dino_swin_backbone import (
    BACKBONE_CHOICES as SWIN_BACKBONE_CHOICES,
    DINOSwinBackbone,
    extract_backbone_state as extract_swin_backbone_state,
    get_model_state as get_swin_model_state,
    load_checkpoint as load_swin_checkpoint,
    resolve_backbone_config,
)

DINO_EVAL_SHORT_SIDE = 800
DINO_EVAL_MAX_SIZE = 1333
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESNET_BACKBONE_CHOICES = ["resnet50", "resnet101"]
BACKBONE_CHOICES = RESNET_BACKBONE_CHOICES + SWIN_BACKBONE_CHOICES
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
        "--clip-dir",
        type=Path,
        default=Path("/workspace/data/tennis/game1/Clip1"),
        help="Directory containing frame images and Label.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Checkpoint path. ResNet backbones expect a trimmed backbone-body "
            "checkpoint by default, while Swin backbones expect the full DINO "
            "checkpoint unless you pass a custom path."
        ),
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=BACKBONE_CHOICES,
        help="Backbone architecture to analyze.",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="Enable dilation in the last ResNet or Swin stage when supported.",
    )
    parser.add_argument(
        "--return-interm-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Intermediate layers returned by the backbone wrapper.",
    )
    parser.add_argument(
        "--pretrain-img-size",
        type=int,
        help="Optional Swin pretraining image size override.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=int,
        default=1,
        help="Minimum visibility value required to include a frame in consistency stats.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for analysis outputs. Defaults depend on the clip and backbone.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading the backbone checkpoint.",
    )
    return parser.parse_args()


def default_checkpoint_path(backbone: str) -> Path:
    if backbone in RESNET_BACKBONE_CHOICES:
        return DINO_WEIGHTS_DIR / "backbone_body_state.pth"
    return DINO_WEIGHTS_DIR / "checkpoint0027_5scale_swin.pth"


def default_output_dir(clip_dir: Path, backbone: str) -> Path:
    return DINO_WEIGHTS_DIR / "analyze" / "output" / f"{clip_dir.name}_{backbone}"


def load_model_for_consistency(
    args: argparse.Namespace,
) -> tuple[nn.Module, nn.modules.module._IncompatibleKeys, dict[str, Any]]:
    checkpoint_path = args.checkpoint or default_checkpoint_path(args.backbone)

    if args.backbone in RESNET_BACKBONE_CHOICES:
        model, load_result = load_backbone_body_state(
            checkpoint_path,
            backbone=args.backbone,
            dilation=args.dilation,
            return_interm_indices=args.return_interm_indices,
            strict=args.strict,
        )
        metadata = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": "trimmed_resnet_backbone",
            "resolved_backbone": args.backbone,
            "resolved_return_interm_indices": list(args.return_interm_indices),
            "resolved_dilation": bool(args.dilation),
        }
        return model, load_result, metadata

    checkpoint = load_swin_checkpoint(checkpoint_path)
    resolved = resolve_backbone_config(
        checkpoint,
        backbone=args.backbone,
        pretrain_img_size=args.pretrain_img_size,
        dilation=args.dilation,
        return_interm_indices=args.return_interm_indices,
        use_checkpoint=None,
    )
    model = DINOSwinBackbone(
        backbone=resolved.backbone,
        pretrain_img_size=resolved.pretrain_img_size,
        dilation=resolved.dilation,
        return_interm_indices=resolved.return_interm_indices,
        use_checkpoint=resolved.use_checkpoint,
    )
    backbone_state = extract_swin_backbone_state(get_swin_model_state(checkpoint))
    load_result = model.backbone.load_state_dict(backbone_state, strict=args.strict)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": "full_swin_dino_checkpoint",
        "resolved_backbone": resolved.backbone,
        "resolved_return_interm_indices": list(resolved.return_interm_indices),
        "resolved_dilation": bool(resolved.dilation),
        "resolved_pretrain_img_size": int(resolved.pretrain_img_size),
        "resolved_use_checkpoint": bool(resolved.use_checkpoint),
    }
    return model, load_result, metadata


def dino_eval_resized_size(image_width: int, image_height: int) -> tuple[int, int]:
    """Return the DINO eval resize output size as (width, height)."""

    size = DINO_EVAL_SHORT_SIDE
    max_size = DINO_EVAL_MAX_SIZE
    min_original_size = float(min((image_width, image_height)))
    max_original_size = float(max((image_width, image_height)))
    if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

    if image_width < image_height:
        out_width = size
        out_height = int(size * image_height / image_width)
    else:
        out_height = size
        out_width = int(size * image_width / image_height)
    return out_width, out_height


def preprocess_like_dino_eval(image_path: Path) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Apply DINO eval preprocessing to one frame."""

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    resized_width, resized_height = dino_eval_resized_size(original_width, original_height)
    resized_image = TVF.resize(image, [resized_height, resized_width])
    tensor = TVF.to_tensor(resized_image)
    tensor = TVF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    meta = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
    }
    image.close()
    resized_image.close()
    return tensor, meta


def sample_feature_at_point(
    feature_map: torch.Tensor,
    x: float,
    y: float,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Sample a feature vector from a feature map at a resized image coordinate."""

    x_norm = (float(x) / float(image_width - 1)) * 2.0 - 1.0
    y_norm = (float(y) / float(image_height - 1)) * 2.0 - 1.0
    grid = torch.tensor([[[[x_norm, y_norm]]]], device=feature_map.device, dtype=feature_map.dtype)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0, 0].detach().cpu()


def cosine_to_previous(features: torch.Tensor) -> list[float | None]:
    values: list[float | None] = [None]
    if features.shape[0] <= 1:
        return values
    normalized = F.normalize(features, dim=1)
    cosines = (normalized[1:] * normalized[:-1]).sum(dim=1)
    values.extend(float(v) for v in cosines)
    return values


def build_frame_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table["frame_name"] = table["file name"].astype(str)
    table["frame_index"] = table["frame_name"].str.replace(".jpg", "", regex=False).astype(int)
    table["x"] = table["x-coordinate"].astype(float)
    table["y"] = table["y-coordinate"].astype(float)
    table["has_coordinates"] = table[["x", "y"]].notna().all(axis=1)
    table["eligible"] = table["has_coordinates"] & (table["visibility"].fillna(-1) >= 0)
    return table.sort_values("frame_index").reset_index(drop=True)


def summarize_scale(
    frame_df: pd.DataFrame,
    feature_tensor: torch.Tensor,
    scale_key: str,
    visibility_threshold: int,
) -> dict[str, Any]:
    valid_mask = frame_df["eligible"] & (
        frame_df["visibility"].fillna(-1) >= visibility_threshold
    )
    valid_rows = frame_df.loc[valid_mask].copy()
    valid_features = feature_tensor[valid_mask.to_numpy(dtype=bool, copy=True)]

    if len(valid_rows) == 0:
        return {
            "scale": scale_key,
            "num_valid_frames": 0,
            "num_consecutive_pairs": 0,
            "mean_feature_norm": None,
            "std_feature_norm": None,
            "mean_cosine_prev": None,
            "std_cosine_prev": None,
            "mean_cosine_prev_gap1": None,
            "std_cosine_prev_gap1": None,
        }

    feature_norms = valid_features.norm(dim=1)
    cosine_prev = cosine_to_previous(valid_features)
    valid_rows["feature_norm"] = feature_norms.tolist()
    valid_rows["cosine_prev"] = cosine_prev
    valid_rows["frame_gap_prev"] = valid_rows["frame_index"].diff()

    gap1 = valid_rows["frame_gap_prev"] == 1
    gap1_cosines = valid_rows.loc[gap1, "cosine_prev"].dropna().astype(float)
    all_cosines = valid_rows["cosine_prev"].dropna().astype(float)

    return {
        "scale": scale_key,
        "num_valid_frames": int(len(valid_rows)),
        "num_consecutive_pairs": int(gap1.sum()),
        "mean_feature_norm": float(feature_norms.mean()),
        "std_feature_norm": float(feature_norms.std(unbiased=False)),
        "mean_cosine_prev": float(all_cosines.mean()) if not all_cosines.empty else None,
        "std_cosine_prev": float(all_cosines.std(ddof=0)) if not all_cosines.empty else None,
        "mean_cosine_prev_gap1": float(gap1_cosines.mean()) if not gap1_cosines.empty else None,
        "std_cosine_prev_gap1": float(gap1_cosines.std(ddof=0)) if not gap1_cosines.empty else None,
    }


def main() -> None:
    args = parse_args()
    clip_dir = args.clip_dir
    label_path = clip_dir / "Label.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    checkpoint_path = args.checkpoint or default_checkpoint_path(args.backbone)
    output_dir = args.output_dir or default_output_dir(clip_dir, args.backbone)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.checkpoint = checkpoint_path
    args.output_dir = output_dir

    model, load_result, load_metadata = load_model_for_consistency(args)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    labels = build_frame_table(pd.read_csv(label_path))
    scale_keys: list[str] | None = None
    per_scale_features: dict[str, list[torch.Tensor]] = {}
    frame_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in labels.itertuples(index=False):
            image_path = clip_dir / row.frame_name
            if not image_path.exists():
                raise FileNotFoundError(f"Missing frame image: {image_path}")

            image_tensor, preprocess_meta = preprocess_like_dino_eval(image_path)
            image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)
            outputs = model(image_tensor)
            if scale_keys is None:
                scale_keys = list(outputs.keys())
                per_scale_features = {str(key): [] for key in scale_keys}

            resized_x = None
            resized_y = None
            if row.eligible:
                resized_x = float(row.x) * float(preprocess_meta["scale_x"])
                resized_y = float(row.y) * float(preprocess_meta["scale_y"])

            record = {
                "frame_name": row.frame_name,
                "frame_index": int(row.frame_index),
                "visibility": None if pd.isna(row.visibility) else int(row.visibility),
                "status": None if pd.isna(row.status) else float(row.status),
                "x": None if pd.isna(row.x) else float(row.x),
                "y": None if pd.isna(row.y) else float(row.y),
                "resized_x": resized_x,
                "resized_y": resized_y,
                "original_width": int(preprocess_meta["original_width"]),
                "original_height": int(preprocess_meta["original_height"]),
                "resized_width": int(preprocess_meta["resized_width"]),
                "resized_height": int(preprocess_meta["resized_height"]),
                "scale_x": float(preprocess_meta["scale_x"]),
                "scale_y": float(preprocess_meta["scale_y"]),
                "has_coordinates": bool(row.has_coordinates),
                "eligible": bool(row.eligible),
            }

            for key in scale_keys:
                feature_map = outputs[key]
                key_str = str(key)
                record[f"scale_{key_str}_shape"] = tuple(feature_map.shape)
                if row.eligible and resized_x is not None and resized_y is not None:
                    feature = sample_feature_at_point(
                        feature_map=feature_map,
                        x=resized_x,
                        y=resized_y,
                        image_width=int(preprocess_meta["resized_width"]),
                        image_height=int(preprocess_meta["resized_height"]),
                    )
                else:
                    feature = torch.full((feature_map.shape[1],), float("nan"))
                per_scale_features[key_str].append(feature)
                record[f"scale_{key_str}_norm"] = None if torch.isnan(feature).any() else float(feature.norm())

            frame_records.append(record)

    assert scale_keys is not None
    frame_df = pd.DataFrame(frame_records).sort_values("frame_index").reset_index(drop=True)
    stacked_features = {
        key: torch.stack(feature_list, dim=0) for key, feature_list in per_scale_features.items()
    }

    per_frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    eligibility_mask = frame_df["eligible"] & (
        frame_df["visibility"].fillna(-1) >= args.visibility_threshold
    )

    for key in scale_keys:
        key_str = str(key)
        features = stacked_features[key_str]
        valid_features = features[eligibility_mask.to_numpy(dtype=bool, copy=True)]
        valid_frame_df = frame_df.loc[eligibility_mask].copy()
        valid_frame_df["feature_norm"] = valid_features.norm(dim=1).tolist()
        valid_frame_df["cosine_prev"] = cosine_to_previous(valid_features)
        valid_frame_df["frame_gap_prev"] = valid_frame_df["frame_index"].diff()
        valid_frame_df["scale"] = key_str
        per_frame_rows.extend(
            valid_frame_df[
                [
                    "scale",
                    "frame_name",
                    "frame_index",
                    "visibility",
                    "status",
                    "x",
                    "y",
                    "resized_x",
                    "resized_y",
                    "feature_norm",
                    "cosine_prev",
                    "frame_gap_prev",
                ]
            ].to_dict(orient="records")
        )
        summary_rows.append(
            summarize_scale(
                frame_df=frame_df,
                feature_tensor=features,
                scale_key=key_str,
                visibility_threshold=args.visibility_threshold,
            )
        )

    feature_payload = {
        "clip_dir": str(clip_dir),
        "checkpoint": str(checkpoint_path),
        "backbone": args.backbone,
        "preprocess": {
            "short_side": DINO_EVAL_SHORT_SIDE,
            "max_size": DINO_EVAL_MAX_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "load_metadata": load_metadata,
        "load_result": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "frame_table": frame_df,
        "scale_keys": [str(k) for k in scale_keys],
        "sampled_features": stacked_features,
    }
    torch.save(feature_payload, output_dir / "ball_features.pt")
    pd.DataFrame(per_frame_rows).to_csv(
        output_dir / "ball_feature_consistency_per_frame.csv",
        index=False,
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ball_feature_consistency_summary.csv", index=False)

    print("Analysis complete")
    print(f"  clip_dir: {clip_dir}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  backbone: {load_metadata['resolved_backbone']}")
    print(f"  device: {device}")
    print(f"  output_dir: {output_dir}")
    print(f"  missing_keys: {list(load_result.missing_keys)}")
    print(f"  unexpected_keys: {list(load_result.unexpected_keys)}")
    print("Scale summary")
    for row in summary_rows:
        print(
            "  scale={scale} valid={num_valid_frames} gap1_pairs={num_consecutive_pairs} "
            "mean_cos_prev_gap1={mean_cosine_prev_gap1}".format(**row)
        )


if __name__ == "__main__":
    main()
