"""Render original-vs-augmented camera-view previews for BLCS training samples.

Usage:
    python -m src.tasks.blcs.scripts.preview_augmentation
    python -m src.tasks.blcs.scripts.preview_augmentation data=singleview_sequence
    python -m src.tasks.blcs.scripts.preview_augmentation preview.split=val preview.max_samples=2
    python -m src.tasks.blcs.scripts.preview_augmentation preview.sample_indices=[0,5,10]

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/preview_augmentation.yaml`.
    - BLCS inputs are abstract 2D observations, so each panel renders the
      camera view: projected court lines plus the ball UV trajectory in
      normalized image coordinates.
    - The base sample is built once per scene with `augment=False` and a
      per-sample-seeded scene RNG, keeping camera selection and the window
      crop identical across rows; each augmented row then applies
      `BLCSBallObservationAugmentation` (all blocks enabled) to that same
      sample under a different torch seed.
    - Rows: original + `preview.num_augmented` draws; columns: up to
      `preview.max_cameras` camera views. A JSON sidecar per sample records
      ball-visibility fractions so dropout effects are quantifiable.
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
from src.tasks.blcs.configuration import PreviewConfig, parse_preview_config
from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.dataset import BallTrajectoryDataset
from src.utils.hydra import hydra_main
from src.utils.io import save_json
from src.utils.rendering.ball_renderer import BallRenderer
from src.utils.rendering.court_renderer import CourtRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.tasks.blcs.data.types import BLCSMultiViewSample

_PANEL_FACECOLOR = "#1a1a1a"
_FIGURE_FACECOLOR = "#101010"


@hydra_main(
    config_path="../configs",
    config_name="preview_augmentation",
    version_base="1.3",
    validation_boundary="blcs.preview_augmentation",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    preview = parse_preview_config(cfg)
    matplotlib.use("Agg")
    output_dir = preview.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = preview.split
    dataset = BallTrajectoryDataset(
        scene_dir=preview.scene_dir,
        split_file=f"{split_name}.txt",
        config=cfg,
        seed=preview.seed,
        augment=False,
    )
    augmentation = BLCSBallObservationAugmentation(
        enable_all_augmentation_blocks(cfg.data.augmentation)
    )

    seed = preview.seed
    num_augmented = preview.num_augmented

    sample_indices = resolve_sample_indices(
        cfg, dataset_size=len(dataset), min_samples=1
    )
    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        # Seed the scene RNG so camera selection and the (center) window draw
        # are reproducible per sample; augmentation draws use torch seeds.
        dataset.rng = np.random.default_rng(seed + sample_index)
        base_sample = dataset[sample_index]

        variants: list[BLCSMultiViewSample] = []
        for variant in range(num_augmented):
            torch.manual_seed(seed + sample_index * 1009 + variant + 1)
            variants.append(augmentation.forward(base_sample))

        scene_name = dataset.scenes[sample_index].stem
        figure = _render_contact_sheet(
            base_sample=base_sample,
            variants=variants,
            scene_name=scene_name,
            config=preview,
        )
        file_stem = f"{sample_index:06d}_{scene_name}"
        image_path = output_dir / f"{file_stem}.png"
        figure.savefig(image_path, dpi=preview.dpi)
        plt.close(figure)

        metadata = {
            "sample_index": sample_index,
            "scene": scene_name,
            "split": split_name,
            "seq_len": int(base_sample["ball_uv"].shape[1]),
            "num_cameras": int(base_sample["ball_uv"].shape[0]),
            "num_augmented": num_augmented,
            "ball_vis_fraction": {
                "original": _visible_fraction(base_sample["ball_vis"]),
                "augmented": [
                    _visible_fraction(variant["ball_vis"]) for variant in variants
                ],
            },
            "output_image": str(image_path),
        }
        save_json(metadata, output_dir / f"{file_stem}.json")
        manifest.append(metadata)

    save_json(manifest, output_dir / "manifest.json")
    print(f"Saved {len(manifest)} augmentation preview(s) to {output_dir}")
    return 0


def _visible_fraction(visibility: torch.Tensor) -> float:
    """Fraction of observations marked visible."""
    return float((visibility > 0.5).float().mean().item())


def _render_contact_sheet(
    *,
    base_sample: BLCSMultiViewSample,
    variants: list[BLCSMultiViewSample],
    scene_name: str,
    config: PreviewConfig,
) -> Figure:
    """Compose a (1 + num_augmented) x num_cameras grid of camera views."""
    num_cameras = min(int(base_sample["ball_uv"].shape[0]), config.max_cameras)
    rows: list[tuple[str, BLCSMultiViewSample]] = [("original", base_sample)]
    rows.extend(
        (f"augmented #{index}", variant) for index, variant in enumerate(variants)
    )

    panel_width = config.panel_width
    panel_height = config.panel_height
    figure, axes = plt.subplots(
        len(rows),
        num_cameras,
        figsize=(panel_width * num_cameras, panel_height * len(rows)),
        squeeze=False,
    )
    figure.patch.set_facecolor(_FIGURE_FACECOLOR)

    court_renderer = CourtRenderer()
    ball_renderer = BallRenderer()
    for row_index, (row_title, sample) in enumerate(rows):
        for camera_index in range(num_cameras):
            ax = axes[row_index][camera_index]
            _render_camera_view(
                ax,
                sample,
                camera_index,
                court_renderer=court_renderer,
                ball_renderer=ball_renderer,
            )
            ax.set_title(f"cam {camera_index} | {row_title}", color="white", fontsize=9)

    seq_len = int(base_sample["ball_uv"].shape[1])
    figure.suptitle(
        f"scene {scene_name} | T={seq_len} | N={int(base_sample['ball_uv'].shape[0])}",
        color="white",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    rendered: Figure = figure
    return rendered


def _render_camera_view(
    ax: Axes,
    sample: BLCSMultiViewSample,
    camera_index: int,
    *,
    court_renderer: CourtRenderer,
    ball_renderer: BallRenderer,
) -> None:
    """Draw one camera view: projected court lines plus the ball UV track."""
    ax.set_facecolor(_PANEL_FACECOLOR)

    # Court keypoints are constant over the window; render frame 0.
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

    ball_uv = sample["ball_uv"][camera_index].numpy()
    ball_vis = sample["ball_vis"][camera_index].numpy() > 0.5
    ball_renderer.render_trajectory_uv(ax, ball_uv, visibility=ball_vis)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#555555")


if __name__ == "__main__":
    sys.exit(main())
