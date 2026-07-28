"""Plot captured and selected camera support without rendering RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.synthetic_data_generation.scene_contract import load_scene_contract
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite output: {args.output}")

    contract = load_scene_contract(args.contract)
    probe = json.loads(args.probe.read_text())
    views = probe.get("views") if isinstance(probe, dict) else None
    if probe.get("status") != "passed" or not isinstance(views, list):
        raise ValueError("Probe is not a passed court_novel_view_probe_v1 artifact.")

    transform = contract.alignment.court_from_scene
    captured_centers = []
    captured_ids = []
    for camera in contract.cameras:
        camera_to_scene = np.asarray(
            camera.camera_to_scene,
            dtype=np.float64,
        ).reshape(4, 4)
        captured_centers.append(transform.apply(camera_to_scene[None, :3, 3])[0])
        captured_ids.append(camera.camera_id)
    captured = np.stack(captured_centers)
    captured_by_id = dict(zip(captured_ids, captured, strict=True))

    novel_matrices = np.stack(
        [
            np.asarray(view["camera_to_court"], dtype=np.float64).reshape(4, 4)
            for view in views
        ]
    )
    novel = novel_matrices[:, :3, 3]
    novel_forwards = novel_matrices[:, :3, 2]
    anchor_ids = {str(view["anchor_camera_id"]) for view in views}
    anchors = np.stack([captured_by_id[value] for value in sorted(anchor_ids)])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    top = axes[0, 0]
    top.plot(captured[:, 0], captured[:, 1], color="0.78", linewidth=1.0)
    top.scatter(captured[:, 0], captured[:, 1], s=7, color="0.62", label="491 SfM")
    top.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=28,
        facecolors="none",
        edgecolors="tab:blue",
        label="42 safe anchors",
    )
    top.scatter(
        novel[:, 0],
        novel[:, 1],
        s=8,
        color="tab:orange",
        alpha=0.75,
        label="256 selected",
    )
    top.quiver(
        novel[::8, 0],
        novel[::8, 1],
        novel_forwards[::8, 0],
        novel_forwards[::8, 1],
        angles="xy",
        scale_units="xy",
        scale=0.25,
        width=0.0025,
        color="tab:red",
        alpha=0.7,
    )
    top.add_patch(
        plt.Rectangle(
            (-HALF_DOUBLES_WIDTH, -HALF_LENGTH),
            2.0 * HALF_DOUBLES_WIDTH,
            2.0 * HALF_LENGTH,
            fill=False,
            color="tab:green",
            linewidth=1.5,
        )
    )
    top.set(title="Court-frame top view", xlabel="court x (m)", ylabel="court y (m)")
    top.set_aspect("equal")
    top.legend(loc="best")
    top.grid(alpha=0.2)

    side = axes[0, 1]
    side.plot(captured[:, 1], captured[:, 2], color="0.78", linewidth=1.0)
    side.scatter(captured[:, 1], captured[:, 2], s=7, color="0.62")
    side.scatter(novel[:, 1], novel[:, 2], s=8, color="tab:orange", alpha=0.75)
    side.quiver(
        novel[::8, 1],
        novel[::8, 2],
        novel_forwards[::8, 1],
        novel_forwards[::8, 2],
        angles="xy",
        scale_units="xy",
        scale=0.25,
        width=0.0025,
        color="tab:red",
        alpha=0.7,
    )
    side.set(
        title="Court-frame side view",
        xlabel="court y (m)",
        ylabel="camera z (m)",
    )
    side.grid(alpha=0.2)

    score = axes[1, 0]
    score.hist(
        [float(view["extrapolation_score"]) for view in views],
        bins=20,
        color="tab:purple",
    )
    score.axvline(1.0, color="black", linestyle="--", label="hard limit")
    score.set(
        title="Nearest-captured coupled support score",
        xlabel="normalized SE(3) score",
        ylabel="selected views",
    )
    score.legend()

    margin = axes[1, 1]
    margin.hist(
        [float(view["min_line_margin_px"]) for view in views],
        bins=20,
        color="tab:green",
    )
    margin.axvline(0.0, color="black", linestyle="--", label="framing limit")
    margin.set(
        title="Minimum projected line-keypoint margin",
        xlabel="pixels",
        ylabel="selected views",
    )
    margin.legend()

    figure.suptitle(
        "B00 safe novel-view probe: local support + hard gates + FVS",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
