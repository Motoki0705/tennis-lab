#!/usr/bin/env python
"""Compare ViT attention (rollout & flow) between two DINOv3 backbones.

This script visualises *what the CLS token looks at* for a vision transformer,
using the two classic propagation methods from Abnar & Zuidema, "Quantifying
Attention Flow in Transformers" (ACL 2020, https://arxiv.org/abs/2005.00928):

* **Attention rollout** — recursively multiplies the per-layer attention
  matrices (after folding in the residual connection as ``0.5*A + 0.5*I`` and
  re-normalising) to obtain an approximate input-to-output attribution. Cheap,
  closed-form.
* **Attention flow** — treats the attention graph as a flow network and takes
  the maximum flow from each input token to the output CLS node. More faithful
  to how information *can* propagate, but needs one max-flow solve per token.

Per-layer attention probabilities are pulled out of the SDPA-based DINOv3
``SelfAttention`` blocks with :class:`AttentionExtractor` (see
``src/utils/models/attention_extraction.py``), since the fused kernel never
returns them.

The script renders, for each input image, a side-by-side panel of the CLS
attention map produced by the tennis-fine-tuned LoRA backbone vs. the original
pretrained backbone, for both methods, and writes the panels under ``outputs/``.

Example::

    .venv/bin/python scripts/analysis/models/attention_maps.py \
        --num-images 4 --image-size 224
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from src.utils.models.attention_extraction import AttentionExtractor
from src.utils.models.loading.dinov3 import (
    DEFAULT_DINOV3_CHECKPOINT,
    DINOv3BackboneAdapter,
    load_dinov3_backbone,
)
from src.utils.paths import resolve_project_path

logger = logging.getLogger("attention_maps")

# DINOv3 / ImageNet normalisation.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_LORA_CHECKPOINT = Path(
    "outputs/dino_ssl/lora_vitb16_20260627/backbone_vitb16.pth"
)
DEFAULT_IMAGE_DIR = Path("data/tennis/dino_ssl/images")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/attention")


# --------------------------------------------------------------------------- #
# Image IO
# --------------------------------------------------------------------------- #
def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Return a normalised ``(1, 3, S, S)`` tensor and the display RGB array."""
    image = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.BICUBIC
    )
    display = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(display).permute(2, 0, 1).clone()
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), display


def select_images(
    image_dir: Path, num_images: int, explicit: list[str] | None
) -> list[Path]:
    """Pick image paths: explicit list if given, else an evenly-spaced sample."""
    if explicit:
        return [resolve_project_path(p) for p in explicit]
    root = resolve_project_path(image_dir)
    candidates = sorted(
        p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        raise FileNotFoundError(f"No images found under {root}")
    # Evenly spaced so we sample across different source clips, not adjacent frames.
    stride = max(1, len(candidates) // num_images)
    return candidates[::stride][:num_images]


# --------------------------------------------------------------------------- #
# Attention capture + propagation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_attentions(
    model: DINOv3BackboneAdapter, pixel_values: torch.Tensor
) -> list[torch.Tensor]:
    """Return per-layer head-fused attention ``(N, N)`` for a single image."""
    with AttentionExtractor(model, fuse_heads=True) as extractor:
        model.forward_features(pixel_values)
    # fuse_heads gives (B, N, N); take the single batch element.
    return [att[0] for att in extractor.attentions]


def _add_residual(attention: torch.Tensor) -> torch.Tensor:
    """Fold in the residual connection: ``0.5*A + 0.5*I`` then row-normalise."""
    identity = torch.eye(attention.size(-1), dtype=attention.dtype)
    augmented = attention + identity
    return augmented / augmented.sum(dim=-1, keepdim=True)


def attention_rollout(
    attentions: list[torch.Tensor], n_special: int, add_residual: bool = True
) -> torch.Tensor:
    """CLS-token attention rollout over patch tokens (Abnar & Zuidema 2020)."""
    result: torch.Tensor | None = None
    for attention in attentions:
        layer = _add_residual(attention) if add_residual else attention
        result = layer if result is None else layer @ result
    assert result is not None
    return result[0, n_special:]  # CLS row -> patch tokens


_FLOW_CAPACITY_SCALE = 100_000  # float capacities -> integers for scipy max-flow.


def attention_flow(
    attentions: list[torch.Tensor],
    n_special: int,
    add_residual: bool = True,
    threshold: float = 0.0,
) -> torch.Tensor:
    """CLS-token attention flow over patch tokens via per-token max-flow.

    Builds a layered DAG whose nodes are ``layer * num_tokens + token`` and
    whose edges carry the (residual-folded) attention weight as capacity, then
    computes the maximum flow from every input token to the final-layer CLS
    node. Uses :func:`scipy.sparse.csgraph.maximum_flow` (C, integer capacities)
    so the per-token solves stay fast; edges below ``threshold`` are pruned.
    """
    layers = [
        (_add_residual(a) if add_residual else a).numpy() for a in attentions
    ]
    depth = len(layers)
    num_tokens = layers[0].shape[0]
    n_nodes = (depth + 1) * num_tokens

    # Edge (layer l, src) -> (layer l+1, dst) with capacity = attn[dst, src].
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    dst_idx, src_idx = np.meshgrid(
        np.arange(num_tokens), np.arange(num_tokens), indexing="ij"
    )
    dst_flat = dst_idx.ravel()
    src_flat = src_idx.ravel()
    for layer_idx, weights in enumerate(layers):
        cap = (weights.ravel() * _FLOW_CAPACITY_SCALE).astype(np.int64)
        keep = cap > int(threshold * _FLOW_CAPACITY_SCALE)
        rows.append(layer_idx * num_tokens + src_flat[keep])
        cols.append((layer_idx + 1) * num_tokens + dst_flat[keep])
        data.append(cap[keep])

    graph = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_nodes, n_nodes),
    )

    sink = depth * num_tokens  # output-layer CLS token (index 0 of last layer).
    flows = np.zeros(num_tokens, dtype=np.float32)
    for token in range(num_tokens):
        result = maximum_flow(graph, token, sink)
        flows[token] = result.flow_value / _FLOW_CAPACITY_SCALE
    return torch.from_numpy(flows[n_special:])


def to_grid_map(cls_attention: torch.Tensor, grid: int, image_size: int) -> np.ndarray:
    """Reshape a per-patch CLS vector to a normalised ``image_size`` heatmap."""
    heat = cls_attention.reshape(1, 1, grid, grid)
    heat = F.interpolate(
        heat, size=(image_size, image_size), mode="bilinear", align_corners=False
    )[0, 0]
    heat = heat - heat.min()
    denom = heat.max().clamp_min(1e-12)
    return (heat / denom).numpy()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_panel(
    display: np.ndarray,
    maps: dict[str, dict[str, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    """Render original + per-(method, model) overlays into one PNG.

    ``maps[method][model_label]`` holds the normalised heatmap.
    """
    methods = list(maps.keys())
    model_labels = list(next(iter(maps.values())).keys())
    n_cols = 1 + len(model_labels)
    n_rows = len(methods)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )
    for r, method in enumerate(methods):
        axes[r][0].imshow(display)
        axes[r][0].set_ylabel(method, fontsize=13)
        axes[r][0].set_title("input" if r == 0 else "")
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
        for c, label in enumerate(model_labels, start=1):
            ax = axes[r][c]
            ax.imshow(display)
            ax.imshow(maps[method][label], cmap="jet", alpha=0.55)
            ax.set_title(label if r == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_models(
    labelled_checkpoints: list[tuple[str, Path]], device: torch.device
) -> dict[str, DINOv3BackboneAdapter]:
    models: dict[str, DINOv3BackboneAdapter] = {}
    for label, ckpt in labelled_checkpoints:
        model = load_dinov3_backbone(checkpoint_path=ckpt)
        model.eval().to(device)
        models[label] = model
        logger.info("loaded %s from %s", label, ckpt)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_LORA_CHECKPOINT)
    parser.add_argument(
        "--pretrained-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT
    )
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional backbone(s) to compare, as 'label=path'. Repeatable.",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--images", nargs="*", default=None, help="Explicit image paths (overrides sampling)."
    )
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument(
        "--image-size", type=int, default=224, help="Square input size (multiple of 16)."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-flow", action="store_true", help="Skip the (slower) attention-flow method."
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.0,
        help="Prune attention edges below this capacity before max-flow.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be a multiple of 16 for patch16.")

    device = torch.device(args.device)
    grid = args.image_size // 16
    n_special = 5  # CLS + 4 storage/register tokens in DINOv3 ViT-B/16.

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labelled: list[tuple[str, Path]] = [
        ("pretrained (LVD-1689M)", resolve_project_path(args.pretrained_checkpoint)),
        ("LoRA tennis-SSL", resolve_project_path(args.lora_checkpoint)),
    ]
    for spec in args.extra_checkpoint:
        if "=" not in spec:
            raise ValueError(f"--extra-checkpoint must be 'label=path', got {spec!r}")
        label, path = spec.split("=", 1)
        labelled.append((label, resolve_project_path(path)))
    models = build_models(labelled, device)
    images = select_images(args.image_dir, args.num_images, args.images)
    logger.info("analysing %d image(s): %s", len(images), [p.name for p in images])

    for image_path in images:
        pixel_values, display = load_image(image_path, args.image_size)
        pixel_values = pixel_values.to(device)

        maps: dict[str, dict[str, np.ndarray]] = {"rollout": {}}
        if not args.no_flow:
            maps["flow"] = {}

        for label, model in models.items():
            attentions = capture_attentions(model, pixel_values)
            rollout = attention_rollout(attentions, n_special)
            maps["rollout"][label] = to_grid_map(rollout, grid, args.image_size)
            if not args.no_flow:
                logger.info("max-flow: %s / %s", image_path.name, label)
                flow = attention_flow(
                    attentions, n_special, threshold=args.flow_threshold
                )
                maps["flow"][label] = to_grid_map(flow, grid, args.image_size)

        out_path = out_dir / f"{image_path.stem}_cls_attention.png"
        render_panel(
            display,
            maps,
            title=f"CLS attention — {image_path.name}",
            out_path=out_path,
        )
        logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
