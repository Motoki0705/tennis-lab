#!/usr/bin/env python3
"""Create compact, Git-trackable visual evidence for the integrated release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CLASS_COLORS = (
    (255, 84, 84),
    (67, 174, 255),
    (255, 194, 64),
    (84, 220, 130),
    (192, 105, 255),
    (0, 210, 210),
    (255, 125, 190),
)
INSTANCE_COLORS = ((235, 74, 64), (37, 140, 240), (255, 186, 0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _title_bar(image: Image.Image, title: str) -> Image.Image:
    bar_height = 28
    canvas = Image.new("RGB", (image.width, image.height + bar_height), "white")
    canvas.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(20, 28, 38), font=ImageFont.load_default())
    return canvas


def _native_mask_panel(
    frame_dir: Path,
    *,
    title: str,
    ball_markers: bool,
) -> Image.Image:
    rgb = Image.open(frame_dir / "rgb.png").convert("RGB")
    array = np.asarray(rgb).copy()
    masks = np.load(frame_dir / "instance_mask.npy", allow_pickle=False)
    if masks.ndim != 3 or masks.shape[:2] != array.shape[:2]:
        raise ValueError(f"Unexpected instance mask: {frame_dir}")
    for channel in range(masks.shape[2]):
        mask = masks[..., channel]
        color = np.asarray(INSTANCE_COLORS[channel % len(INSTANCE_COLORS)])
        array[mask] = (0.35 * array[mask] + 0.65 * color).astype(np.uint8)
    panel = Image.fromarray(array)
    draw = ImageDraw.Draw(panel)
    for channel in range(masks.shape[2]):
        ys, xs = np.where(masks[..., channel])
        if len(xs) == 0:
            continue
        color = INSTANCE_COLORS[channel % len(INSTANCE_COLORS)]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        padding = 10 if ball_markers else 2
        draw.rectangle(
            (x0 - padding, y0 - padding, x1 + padding, y1 + padding),
            outline=color,
            width=3,
        )
        if ball_markers:
            cx, cy = int(xs.mean()), int(ys.mean())
            draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), outline=color, width=3)
            draw.text(
                (cx + 16, max(2, cy - 8)),
                f"ball {channel + 1}",
                fill=color,
                font=ImageFont.load_default(),
                stroke_width=2,
                stroke_fill="white",
            )
    return _title_bar(panel, title)


def _court_panel(frame_dir: Path, *, title: str) -> Image.Image:
    panel = Image.open(frame_dir / "rgb.png").convert("RGB")
    labels = _read_json(frame_dir / "labels.json")
    draw = ImageDraw.Draw(panel)
    for court_index, court in enumerate(labels["projection"]["courts"]):
        for class_record in court["classes"]:
            color = CLASS_COLORS[class_record["class_id"]]
            for point in class_record["points"]:
                if not point["in_frame"]:
                    continue
                x, y = (int(round(value)) for value in point["uv"])
                radius = 5
                box = (x - radius, y - radius, x + radius, y + radius)
                if point["visible"]:
                    if court_index % 2 == 0:
                        draw.ellipse(box, fill=color, outline="black", width=1)
                    else:
                        draw.rectangle(box, fill=color, outline="black", width=1)
                else:
                    draw.ellipse(box, outline=color, width=2)
    return _title_bar(panel, title)


def _save_gif(frames: list[Image.Image], path: Path, *, duration_ms: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty GIF.")
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    normalized = []
    for frame in frames:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        normalized.append(canvas.quantize(colors=128))
    normalized[0].save(
        path,
        save_all=True,
        append_images=normalized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _save_overview(
    output: Path,
    *,
    blcs_single: Image.Image,
    blcs_multi: Image.Image,
    plcs_multi: Image.Image,
    court: Image.Image,
) -> None:
    panels = (blcs_single, blcs_multi, plcs_multi, court)
    titles = (
        "BLCS single — native RGB + exact diagnostic mask",
        "BLCS multi — two persistent ball instances",
        "PLCS multi — SMPL-X-controlled Gaussian people",
        "Court — two instances, seven symmetric classes",
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, panel, title in zip(axes.flat, panels, titles, strict=True):
        axis.imshow(panel)
        axis.set_title(title, fontsize=11, loc="left", weight="bold")
        axis.axis("off")
    fig.suptitle(
        "3DGS-native synthetic data: one scene, movable Gaussians, exact labels",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def _plot_court(ax: Any) -> None:
    ax.plot(
        [-4.115, 4.115, 4.115, -4.115, -4.115],
        [-11.885, -11.885, 11.885, 11.885, -11.885],
        color="#303944",
        linewidth=1,
    )
    ax.plot([-4.115, 4.115], [0, 0], color="#303944", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("court x (m)")
    ax.set_ylabel("court y (m)")
    ax.grid(alpha=0.2)


def _plot_track(
    ax: Any,
    positions: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    for instance in range(positions.shape[1]):
        ax.plot(
            positions[:, instance, 0],
            positions[:, instance, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.9,
            label=label if instance == 0 else None,
        )


def _save_trajectory_plot(repo: Path, output: Path) -> None:
    artifacts = repo / ".codex-loop/3dgs-synthetic-data/artifacts"
    blcs_a = np.load(
        artifacts
        / "cycle-09/prototype-multi-plan-v1/plan/positions_court_m.npy",
        allow_pickle=False,
    )
    blcs_b = np.load(
        artifacts
        / "cycle-16/blcs-multi-distinct-seed-v1/plan/positions_court_m.npy",
        allow_pickle=False,
    )
    plcs_a = np.load(
        artifacts / "cycle-12/plcs-multi-plan-v1/positions_court_m.npy",
        allow_pickle=False,
    )
    plcs_b = np.load(
        artifacts / "cycle-16/plcs-multi-distinct-seed-v1/positions_court_m.npy",
        allow_pickle=False,
    )
    orbit_a = _read_json(
        artifacts / "cycle-14/multicourt-orbit-plan-v1/manifest.json"
    )
    orbit_b = _read_json(
        artifacts / "cycle-16/court-orbit-distinct-seed-v1/manifest.json"
    )
    contract = _read_json(
        Path(
            "/home/kamimura/projects/tennis-lab/data/tennis/3dgs_scenes/"
            "b00-default-v1/scene-contract-ground-line-user-override-v2.json"
        )
    )

    def centers(manifest: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                np.asarray(frame["camera"]["camera_to_scene"]).reshape(4, 4)[:3, 3]
                for frame in manifest["frames"]
            ]
        )

    captured = np.asarray(
        [
            np.asarray(camera["camera_to_scene"]).reshape(4, 4)[:3, 3]
            for camera in contract["cameras"]
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    _plot_court(axes[0])
    _plot_track(
        axes[0],
        blcs_a,
        label="accepted seed",
        color="#e0473e",
        linestyle="-",
    )
    _plot_track(
        axes[0],
        blcs_b,
        label="distinct seed",
        color="#2878c8",
        linestyle="--",
    )
    axes[0].set_title("BLCS physics trajectories", loc="left", weight="bold")
    axes[0].legend(fontsize=8)

    _plot_court(axes[1])
    _plot_track(
        axes[1],
        plcs_a,
        label="accepted seed",
        color="#e0473e",
        linestyle="-",
    )
    _plot_track(
        axes[1],
        plcs_b,
        label="distinct seed",
        color="#2878c8",
        linestyle="--",
    )
    axes[1].set_title("PLCS person placement", loc="left", weight="bold")
    axes[1].legend(fontsize=8)

    center_a = centers(orbit_a)
    center_b = centers(orbit_b)
    axes[2].scatter(
        captured[:, 0],
        captured[:, 1],
        s=7,
        alpha=0.35,
        color="#5f6872",
        label="491 SfM cameras",
    )
    axes[2].scatter(
        center_a[:, 0],
        center_a[:, 1],
        s=5,
        alpha=0.45,
        color="#e0473e",
        label="accepted orbit seed",
    )
    axes[2].scatter(
        center_b[:, 0],
        center_b[:, 1],
        s=5,
        alpha=0.35,
        color="#2878c8",
        label="distinct orbit seed",
    )
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("scene x")
    axes[2].set_ylabel("scene y")
    axes[2].grid(alpha=0.2)
    axes[2].set_title("SfM envelope → bold orbit families", loc="left", weight="bold")
    axes[2].legend(fontsize=8)
    fig.suptitle("Measured seed diversity and camera expansion", fontsize=16, weight="bold")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def _save_metric_plot(repo: Path, output: Path) -> None:
    report = _read_json(
        repo
        / ".codex-loop/3dgs-synthetic-data/artifacts/cycle-16/"
        "p8-acceptance-v2/report.json"
    )
    p3 = _read_json(
        repo
        / ".codex-loop/3dgs-synthetic-data/artifacts/cycle-09/"
        "p3-acceptance-report-v1.json"
    )
    p5 = _read_json(
        repo
        / ".codex-loop/3dgs-synthetic-data/artifacts/cycle-12/"
        "p5-acceptance-report-v2.json"
    )
    p7 = _read_json(
        repo
        / ".codex-loop/3dgs-synthetic-data/artifacts/cycle-15/"
        "p7-acceptance-v2/report.json"
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    repeat_names = list(report["same_seed"])
    repeat_values = [
        report["same_seed"][name]["file_count"] for name in repeat_names
    ]
    axes[0].barh(
        range(len(repeat_names)),
        repeat_values,
        color="#3ba272",
    )
    axes[0].set_yticks(range(len(repeat_names)), repeat_names)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("byte-identical files (log scale)")
    axes[0].set_title("Same-seed reproducibility", loc="left", weight="bold")
    for index, value in enumerate(repeat_values):
        axes[0].text(value * 1.04, index, str(value), va="center", fontsize=8)

    scale_names = ("BLCS\nball Gaussians", "PLCS\navatar Gaussians", "Court\nframes")
    scale_values = (
        p3["prototype_asset"]["gaussian_count"],
        p5["multi"]["person_count"] * 4096,
        p7["metrics"]["frame_count"],
    )
    bars = axes[1].bar(scale_names, scale_values, color=("#e0473e", "#2878c8", "#b66ad2"))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("count (log scale)")
    axes[1].set_title("Accepted release scale", loc="left", weight="bold")
    axes[1].bar_label(bars, fmt="%d", padding=3)

    gate_names = list(report["gates"])
    axes[2].barh(
        range(len(gate_names)),
        [1] * len(gate_names),
        color="#3ba272",
    )
    axes[2].set_yticks(range(len(gate_names)), gate_names, fontsize=7)
    axes[2].set_xlim(0, 1.06)
    axes[2].set_xticks((0, 1), ("fail", "pass"))
    axes[2].set_title(
        f"Integrated gate: {sum(report['gates'].values())}/{len(gate_names)}",
        loc="left",
        weight="bold",
    )
    fig.suptitle("Release metrics: exact repeats, scale, and acceptance", fontsize=16, weight="bold")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def _copy_scaled(source: Path, destination: Path, *, max_width: int = 1800) -> None:
    image = Image.open(source).convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    image.save(destination, optimize=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    artifacts = repo / ".codex-loop/3dgs-synthetic-data/artifacts"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        blcs_single_dirs = sorted(
            (artifacts / "cycle-09/prototype-single-render-v3/frames").iterdir()
        )
        blcs_multi_dirs = sorted(
            (artifacts / "cycle-09/prototype-multi-render-v2/frames").iterdir()
        )
        plcs_single_dirs = sorted(
            (artifacts / "cycle-12/plcs-single-render-v1/frames").iterdir()
        )
        plcs_multi_dirs = sorted(
            (artifacts / "cycle-12/plcs-multi-render-v1/frames").iterdir()
        )
        court_manifest = _read_json(
            artifacts / "cycle-15/court-dataset-v1/manifest.json"
        )
        court_entries = court_manifest["frames"][::36]
        court_dirs = [
            (
                artifacts
                / "cycle-15/court-dataset-v1"
                / Path(entry["rgb"]["relative_path"]).parent
            )
            for entry in court_entries
        ]

        blcs_single_frames = [
            _native_mask_panel(
                frame,
                title=f"BLCS single · {frame.name} · exact AOV mask",
                ball_markers=True,
            )
            for frame in blcs_single_dirs
        ]
        blcs_multi_frames = [
            _native_mask_panel(
                frame,
                title=f"BLCS multi · {frame.name} · red/blue identity",
                ball_markers=True,
            )
            for frame in blcs_multi_dirs
        ]
        plcs_single_frames = [
            _native_mask_panel(
                frame,
                title=f"PLCS single · {frame.name} · controlled Gaussian avatar",
                ball_markers=False,
            )
            for frame in plcs_single_dirs
        ]
        plcs_multi_frames = [
            _native_mask_panel(
                frame,
                title=f"PLCS multi · {frame.name} · persistent red/blue identity",
                ball_markers=False,
            )
            for frame in plcs_multi_dirs
        ]
        court_frames = [
            _court_panel(
                frame,
                title=(
                    f"Court · {entry['split']} · {entry['family_id']} · "
                    "circle=court_0 square=court_1"
                ),
            )
            for frame, entry in zip(court_dirs, court_entries, strict=True)
        ]
        _save_gif(
            blcs_single_frames + blcs_multi_frames,
            temporary / "blcs-native-labels.gif",
            duration_ms=650,
        )
        _save_gif(
            plcs_single_frames + plcs_multi_frames,
            temporary / "plcs-native-labels.gif",
            duration_ms=650,
        )
        _save_gif(
            court_frames,
            temporary / "court-native-labels.gif",
            duration_ms=600,
        )
        _save_overview(
            temporary / "native-composition-overview.png",
            blcs_single=blcs_single_frames[0],
            blcs_multi=blcs_multi_frames[0],
            plcs_multi=plcs_multi_frames[0],
            court=court_frames[0],
        )
        _save_trajectory_plot(repo, temporary / "seed-diversity-trajectories.png")
        _save_metric_plot(repo, temporary / "release-metrics.png")
        _copy_scaled(
            artifacts / "cycle-12/plcs-multi-diagnostic-v1.png",
            temporary / "plcs-pose-contact-sheet.png",
        )
        _copy_scaled(
            artifacts
            / "cycle-15/court-dataset-diagnostic-v2/label-contact-sheet.png",
            temporary / "court-keypoint-contact-sheet.png",
        )
        _copy_scaled(
            artifacts
            / "cycle-15/court-dataset-diagnostic-v2/heatmap-contact-sheet.png",
            temporary / "court-heatmap-contact-sheet.png",
        )
        files = {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema": "tennis_3dgs_native_release_visuals_v1",
            "status": "passed",
            "scope": (
                "diagnostic-only label overlays; accepted dataset RGB remains unchanged"
            ),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
