"""Render the fixed numerical comparison; no training data or real ball labels."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rows = json.loads((root / "comparison.json").read_text())["summary"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
scenarios = ["unimodal", "ambiguous", "low_presence", "prior_only"]
x = np.arange(4)
for i, method in enumerate(["A", "B", "C"]):
    selected = [
        next(r for r in rows if r["scenario"] == s and r["method"] == method)
        for s in scenarios
    ]
    for ax, metric in zip(
        axes, ["mean_nll_m3", "coverage95", "median_ms"], strict=True
    ):
        ax.bar(
            x + (i - 1) * 0.24, [r[metric] for r in selected], width=0.24, label=method
        )
for ax in axes:
    ax.set_xticks(x, ["Single", "Two modes", "Low presence", "Prior"], rotation=20)
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("3D NLL (nat; density in m^-3)")
axes[1].set_ylabel("Empirical 95% HDR coverage")
axes[1].set_ylim(0.8, 1.02)
axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1)
axes[2].set_ylabel("Median CPU ms/frame (log scale)")
axes[2].set_yscale("log")
axes[2].legend(title="Method")
fig.suptitle("Synthetic comparison: 128 trials/condition, Meiji camera geometry only")
fig.savefig(root / "comparison.png", dpi=150)
