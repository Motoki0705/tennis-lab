"""CPU-only summaries and comparisons on the same held-out image names."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .validation import validate_ready
from .workspace import load_manifest, sha256, write_json


def common_metrics(evaluations: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("At least one evaluation is required")
    names = sorted(set.intersection(*(set(run) for run in evaluations)))
    if not names:
        raise ValueError("Runs have no common validation images")
    metrics = ("psnr", "ssim", "lpips")
    means = []
    for run in evaluations:
        if any(
            not math.isfinite(run[name][metric]) for name in names for metric in metrics
        ):
            raise ValueError("Non-finite common validation metric")
        means.append(
            {
                metric: sum(run[name][metric] for name in names) / len(names)
                for metric in metrics
            }
        )
    return {"image_names": names, "means": means}


def write_training_curves(root: Path, output: Path) -> None:
    """Plot NHT's actual scalar tags, which differ from Lightning task tags."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(root / "reconstruction/3dgs/model/tb"), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tag in zip(axes, ("train/loss", "train/num_GS"), strict=True):
        values = accumulator.Scalars(tag)
        if not values:
            raise ValueError(f"Missing TensorBoard samples: {tag}")
        ax.plot([item.step for item in values], [item.value for item in values])
        ax.set(title=tag, xlabel="Training step")
        ax.grid(alpha=0.25)
    figure.suptitle(root.name)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def compare_training_runs(roots: list[Path], output: Path) -> dict[str, Any]:
    roots = [root.resolve() for root in roots]
    if len(roots) != len(set(roots)) or not roots:
        raise ValueError("Provide distinct completed run roots")
    evaluations = []
    renders = []
    observed = []
    reports = []
    for root in roots:
        manifest = load_manifest(root)
        validate_ready(root, manifest)
        if manifest.status != "complete":
            raise ValueError(f"Run is not complete: {root}")
        workspace = root / "reconstruction"
        training = json.loads((workspace / "3dgs/training.json").read_text())
        metadata = json.loads((workspace / "3dgs/scene-metadata.json").read_text())
        cameras = [
            camera for camera in metadata["cameras"] if camera["split"] == "validation"
        ]
        step = training["max_steps"] - 1
        per_image = json.loads(
            (workspace / f"3dgs/model/stats/val_step{step}_per_image.json").read_text()
        )
        if any(len(values) != len(cameras) for values in per_image.values()):
            raise ValueError(f"Per-image metrics and validation names differ: {root}")
        evaluations.append(
            {
                camera["image_name"]: {
                    key: float(values[i]) for key, values in per_image.items()
                }
                for i, camera in enumerate(cameras)
            }
        )
        render_paths = {
            camera["image_name"]: workspace
            / f"3dgs/model/renders/val_step{step}_{i:04d}.png"
            for i, camera in enumerate(cameras)
        }
        if not all(path.is_file() for path in render_paths.values()):
            raise ValueError(f"Missing validation renders: {root}")
        renders.append(render_paths)
        observed.append(
            {
                camera["image_name"]: sha256(
                    workspace / "3dgs/observed-images" / camera["observed_image"]
                )
                for camera in cameras
            }
        )
        reports.append(
            {
                "root": str(root),
                "scene_id": manifest.config.scene_id,
                "images": len(manifest.frames),
                "training_steps": training["max_steps"],
                "validation_images": len(cameras),
                "all_validation_metrics": training["validation_metrics"][-1]["metrics"],
                "elapsed_seconds": training["elapsed_seconds"],
            }
        )
    common = common_metrics(evaluations)
    names = common["image_names"]
    if any(len({run[name] for run in observed}) != 1 for name in names):
        raise ValueError("Common validation target pixels differ between runs")
    for report, means in zip(reports, common["means"], strict=True):
        report["common_validation_metrics"] = means
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "common_validation_names": names,
        "common_target_pixels_identical": True,
        "metric_source": "NHT built-in validation (one optimizer update after its saved checkpoint)",
        "runs": reports,
    }
    write_json(output / "comparison.json", summary)
    cards = []
    width, height, header = 480, 270, 30
    for page in range((len(names) + 3) // 4):
        group = names[page * 4 : (page + 1) * 4]
        sheet = Image.new(
            "RGB", (width * (len(roots) + 1), (height + header) * len(group)), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for row, name in enumerate(group):
            panels = []
            for i, run in enumerate(renders):
                with Image.open(run[name]) as image:
                    mid = image.width // 2
                    if i == 0:
                        panels.append(image.crop((0, 0, mid, image.height)))
                    panels.append(image.crop((mid, 0, image.width, image.height)))
            labels = [
                "Generated target",
                *[
                    f"{run['images']} images / {run['training_steps']} steps"
                    for run in reports
                ],
            ]
            for col, (panel, label) in enumerate(zip(panels, labels, strict=True)):
                sheet.paste(
                    panel.resize((width, height), Image.Resampling.LANCZOS),
                    (col * width, row * (height + header) + header),
                )
                draw.text(
                    (col * width + 5, row * (height + header) + 8),
                    f"{name} | {label}",
                    fill="black",
                )
        path = output / f"common-validation-{page + 1:02d}.jpg"
        sheet.save(path, quality=95)
        cards.append(
            f'<img src="{html.escape(path.name)}" alt="Common validation comparison">'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>B00 appearance experiment comparison</title><style>body{font:15px sans-serif;margin:20px}img{width:100%;margin-bottom:24px}pre{white-space:pre-wrap}</style><h1>Common validation views</h1><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>"
        + "".join(cards),
        encoding="utf-8",
    )
    return summary
