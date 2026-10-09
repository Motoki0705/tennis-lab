"""Select one CNN only after comparable pretraining runs have all completed."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    pretraining_config,
)

from .checkpoint import completed_pretraining

SHARED_RECIPE = ("epochs", "windows_per_epoch", "batch_size", "seed", "learning_rate", "warmup_updates",
                 "schedule", "weight_decay", "gradient_clip", "manifest_sha256", "selection_scope", "selection_metric",
                 "sigma_ratio", "focal_gamma", "input_contract", "image_decode", "supervision", "test_usage")
SHARED_MODEL = ("stem_channels", "mixed_channels", "residual_blocks", "decoder_channels", "dim", "heads", "layers", "ffn_dim")


def compare_pretraining(runs: dict[str, Path], manifest: Path) -> dict[str, Any]:
    if set(runs) != {"residual", "convnext_v2", "fasternet"}:
        raise ValueError("The comparison requires exactly the declared three CNNs")
    rows = []
    shared: dict[str, Any] | None = None
    for variant, run in runs.items():
        path, checkpoint = completed_pretraining(run, manifest)
        config = pretraining_config(checkpoint)
        if config.encoder_variant != variant:
            raise ValueError("Run directory contains the wrong CNN architecture")
        recipe = checkpoint["recipe"]
        if recipe["selection_scope"] != "common" or recipe["test_usage"] != "none":
            raise ValueError("CNN selection requires common validation and no test usage")
        signature = {key: recipe[key] for key in SHARED_RECIPE}
        signature["model"] = {key: getattr(config, key) for key in SHARED_MODEL}
        signature["precision"] = recipe["runtime"]["precision"]
        signature["compile_mode"] = recipe["runtime"]["compile_mode"]
        if shared is not None and signature != shared:
            raise ValueError("Pretraining data/budget/precision or DPT contract differs between CNNs")
        shared = signature
        completed = json.loads((run / "COMPLETED.json").read_text())
        expected = recipe["epochs"] * math.ceil(recipe["windows_per_epoch"] / recipe["batch_size"])
        if completed["global_step"] != expected:
            raise ValueError("Pretraining did not complete its declared update budget")
        report = checkpoint["validation"]["scopes"]
        common, full = report["common"]["macro_mean_error_px"], report["full"]["macro_mean_error_px"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (common, full)):
            raise ValueError("CNN selection requires finite complete-FPS validation metrics")
        logs = [json.loads(s) for s in (run / "train.jsonl").read_text().splitlines()]
        epoch0 = [r for r in logs if r["epoch"] == 0]
        rate = ((epoch0[-1]["windows"] - epoch0[0]["windows"]) /
                (epoch0[-1]["train_seconds"] - epoch0[0]["train_seconds"])) if len(epoch0) > 1 else None
        rows.append(dict(variant=variant, run=str(run), checkpoint=str(path), epoch=checkpoint["epoch"],
            common_error_px=common, full_error_px=full, total_updates=completed["global_step"],
            parameters=sum(v.numel() for v in checkpoint["state_dict"].values() if isinstance(v, torch.Tensor)),
            steady_epoch0_windows_per_second=rate,
            peak_allocated_gib=max((r.get("peak_allocated_gib", 0.) for r in logs), default=0.)))
    rows.sort(key=lambda r: (r["common_error_px"], r["full_error_px"], r["variant"]))
    return dict(schema="mdd_cnn_comparison.v1", selection="minimum common error; full error then name break ties",
                shared=shared, ranking=rows, winner=rows[0]["variant"], selected_run=rows[0]["run"],
                limitation="single seed; speed excludes only first logged block; not isolated block ablation")


def save_comparison(report: dict[str, Any], directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    rows = report["ranking"]
    names = [r["variant"] for r in rows]
    axes[0].barh(names, [r["common_error_px"] for r in rows], color=["#147d92", "#638bb4", "#9fb5c7"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Common validation error (source pixels; lower is better)", title="Pretraining accuracy")
    for i, row in enumerate(rows):
        axes[0].text(row["common_error_px"], i, f"  {row['common_error_px']:.2f}", va="center")
        rate = row["steady_epoch0_windows_per_second"]
        if rate is not None:
            axes[1].scatter(rate, row["common_error_px"], s=95, color=f"C{i}")
            axes[1].annotate(row["variant"], (rate, row["common_error_px"]), xytext=(5, 7), textcoords="offset points")
    axes[1].set(xlabel="Training windows / second (higher is better)", ylabel="Common error (pixels)", title="Accuracy and measured throughput")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Three CNNs | Same DPT, frozen data, seed and update budget", fontweight="bold")
    fig.supxlabel("Single-seed experiment. Speed omits the first logged block; hardware load can vary. Select by accuracy, not FLOPs.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
