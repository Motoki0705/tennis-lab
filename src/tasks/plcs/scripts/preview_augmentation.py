"""Render original-vs-augmented camera-view previews for PLCS training samples.

Usage:
    python -m src.tasks.plcs.scripts.preview_augmentation
    python -m src.tasks.plcs.scripts.preview_augmentation data=singleview_sequence
    python -m src.tasks.plcs.scripts.preview_augmentation preview.split=val preview.max_samples=2
    python -m src.tasks.plcs.scripts.preview_augmentation preview.sample_indices=[0,5,10]

Notes:
    - Hydra loads configuration from `src/tasks/plcs/configs/preview_augmentation.yaml`.
    - PLCS inputs are abstract 2D observations, so each panel renders the
      camera view: projected court lines plus a fading COCO17 skeleton trail
      (`preview.pose_frames` snapshots) in normalized image coordinates.
    - The base sample is built once per scene with `augment=False` and a
      per-sample-seeded scene RNG, keeping camera selection and the window
      crop identical across rows; each augmented row then applies
      `PLCSObservationAugmentation` (all blocks enabled) to that same sample
      under a different torch seed.
    - Rows: original + `preview.num_augmented` draws; columns: up to
      `preview.max_cameras` camera views. A JSON sidecar per sample records
      human/court visibility fractions so dropout effects are quantifiable.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.base.visualization.preview import (
    enable_all_augmentation_blocks,
    resolve_sample_indices,
)
from src.tasks.plcs.configuration import PLCSPreviewRuntimeConfig
from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.tasks.plcs.data.dataset import SceneDataset
from src.utils.hydra import hydra_main
from src.utils.io import save_json
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor

_PANEL_FACECOLOR = "#1a1a1a"
_FIGURE_FACECOLOR = "#101010"


@hydra_main(
    config_path="../configs",
    config_name="preview_augmentation",
    version_base="1.3",
    validation_boundary="plcs.preview_augmentation",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    matplotlib.use("Agg")
    runtime = PLCSPreviewRuntimeConfig.from_config(cfg)
    output_dir = runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = str(cfg.preview.split)
    dataset = SceneDataset(
        scene_dir=runtime.scene_dir,
        split_file=f"{split_name}.txt",
        config=cfg,
        seed=int(cfg.preview.seed),
        augment=False,
    )
    augmentation = PLCSObservationAugmentation(
        enable_all_augmentation_blocks(cfg.data.augmentation)
    )

    seed = int(cfg.preview.seed)
    num_augmented = int(cfg.preview.num_augmented)
    if num_augmented < 1:
        raise ValueError("preview.num_augmented must be >= 1.")

    sample_indices = resolve_sample_indices(
        cfg, dataset_size=len(dataset), min_samples=1
    )
    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        # Seed the scene RNG so camera selection and the (center) window draw
        # are reproducible per sample; augmentation draws use torch seeds.
        dataset.rng = np.random.default_rng(seed + sample_index)
        base_sample = dataset[sample_index]

        variants: list[dict[str, Tensor]] = []
        for variant in range(num_augmented):
            torch.manual_seed(seed + sample_index * 1009 + variant + 1)
            variants.append(augmentation.forward(base_sample))

        scene_name = dataset.scenes[sample_index].stem
        figure = _render_contact_sheet(
            base_sample=base_sample,
            variants=variants,
            scene_name=scene_name,
            cfg=cfg,
        )
        file_stem = f"{sample_index:06d}_{scene_name}"
        image_path = output_dir / f"{file_stem}.png"
        figure.savefig(image_path, dpi=int(cfg.preview.figure.dpi))
        plt.close(figure)

        metadata = {
            "sample_index": sample_index,
            "scene": scene_name,
            "split": split_name,
            "seq_len": int(base_sample["human_kp"].shape[1]),
            "num_cameras": int(base_sample["human_kp"].shape[0]),
            "num_augmented": num_augmented,
            "human_visible_fraction": {
                "original": _visible_fraction(base_sample["human_vis"]),
                "augmented": [
                    _visible_fraction(variant["human_vis"]) for variant in variants
                ],
            },
            "court_visible_fraction": {
                "original": _visible_fraction(base_sample["court_vis"]),
                "augmented": [
                    _visible_fraction(variant["court_vis"]) for variant in variants
                ],
            },
            "output_image": str(image_path),
        }
        save_json(metadata, output_dir / f"{file_stem}.json")
        manifest.append(metadata)

    save_json(manifest, output_dir / "manifest.json")
    print(f"Saved {len(manifest)} augmentation preview(s) to {output_dir}")
    return 0


def _visible_fraction(visibility: Tensor) -> float:
    """Fraction of observations marked visible."""
    return float((visibility > 0.5).float().mean().item())


def _render_contact_sheet(
    *,
    base_sample: dict[str, Tensor],
    variants: list[dict[str, Tensor]],
    scene_name: str,
    cfg: DictConfig,
) -> Figure:
    """Compose a (1 + num_augmented) x num_cameras grid of camera views."""
    num_cameras = min(
        int(base_sample["human_kp"].shape[0]), int(cfg.preview.max_cameras)
    )
    rows: list[tuple[str, dict[str, Tensor]]] = [("original", base_sample)]
    rows.extend(
        (f"augmented #{index}", variant) for index, variant in enumerate(variants)
    )

    panel_width = float(cfg.preview.figure.panel_width)
    panel_height = float(cfg.preview.figure.panel_height)
    figure, axes = plt.subplots(
        len(rows),
        num_cameras,
        figsize=(panel_width * num_cameras, panel_height * len(rows)),
        squeeze=False,
    )
    figure.patch.set_facecolor(_FIGURE_FACECOLOR)

    court_renderer = CourtRenderer()
    skeleton_renderer = SkeletonRenderer(skeleton_type="coco17")
    pose_frames = int(cfg.preview.pose_frames)
    for row_index, (row_title, sample) in enumerate(rows):
        for camera_index in range(num_cameras):
            ax = axes[row_index][camera_index]
            _render_camera_view(
                ax,
                sample,
                camera_index,
                court_renderer=court_renderer,
                skeleton_renderer=skeleton_renderer,
                pose_frames=pose_frames,
            )
            ax.set_title(f"cam {camera_index} | {row_title}", color="white", fontsize=9)

    seq_len = int(base_sample["human_kp"].shape[1])
    figure.suptitle(
        f"scene {scene_name} | T={seq_len} | N={int(base_sample['human_kp'].shape[0])}",
        color="white",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    rendered: Figure = figure
    return rendered


def _render_camera_view(
    ax: Axes,
    sample: dict[str, Tensor],
    camera_index: int,
    *,
    court_renderer: CourtRenderer,
    skeleton_renderer: SkeletonRenderer,
    pose_frames: int,
) -> None:
    """Draw one camera view: projected court lines plus a skeleton trail."""
    ax.set_facecolor(_PANEL_FACECOLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip Y for image coordinates.

    # Court keypoints are near-constant over the window; render frame 0.
    court_kp = sample["court_kp"][camera_index, 0].numpy()
    court_vis = sample["court_vis"][camera_index, 0].numpy() > 0.5
    court_renderer.render_projected_2d(
        ax,
        court_kp,
        court_vis,
        line_color="lime",
        line_width=1.0,
        visible_line_alpha=0.5,
        partial_line_alpha=0.2,
        keypoint_color="lime",
        keypoint_size=18.0,
        keypoint_alpha=0.7,
    )

    human_kp = sample["human_kp"][camera_index].numpy()
    human_vis = sample["human_vis"][camera_index].numpy() > 0.5
    skeleton_renderer.render_sequence_2d(
        ax,
        human_kp,
        human_vis,
        num_frames=min(pose_frames, human_kp.shape[0]),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#555555")


if __name__ == "__main__":
    sys.exit(main())
