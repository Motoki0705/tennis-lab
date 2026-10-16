"""Export the recorded long-reader comparison. No training is executed."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
old = json.loads((ROOT.parent / "run-i986-nvjpeg-long-reader-20261009/report.json").read_text())
middle = json.loads((ROOT.parent / "run-i986-pread-long-20261009/report.json").read_text())
new = json.loads((ROOT / "async-w8-p4-b1.json").read_text())
ink, muted = "#172d3e", "#526777"
colors = ["#ba574d", "#c19542", "#008b80"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink,
                     "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
fig.subplots_adjust(left=.20, right=.95, bottom=.19, top=.79, wspace=.43, hspace=.78)
fig.text(.055, .945, "Keeping the GPU supplied across 1,020 windows", size=21, weight="bold")
fig.text(.055, .902, "Same sampled windows and loss sequence  |  BF16  |  Conv2d + query-only  |  BS1", color=muted)
fig.text(.055, .865, "1,008 measured updates after warmup. Pread trials requested eviction of clean input-file pages.", size=10, color=muted)
labels = ["mmap / 2w × 2", "pread / 2w × 2", "pread / 8w × 4"]
y = [2, 1, 0]

ax = axes[0, 0]
values = [d["windows_per_second"] for d in (old, middle, new)]
ax.barh(y, values, color=colors, height=.54)
ax.set_yticks(y, labels)
ax.set_xlim(0, 17)
ax.set_xlabel("Windows / second  (higher is faster)")
ax.set_title("A  Sustained throughput", loc="left", weight="bold", pad=14)
for pos, value in zip(y, values, strict=True):
    ax.text(value + .20, pos, f"{value:.2f}", va="center", weight="bold")

ax = axes[0, 1]
waits = [d["mean_reader_wait_seconds"] * 1000 for d in (old, middle, new)]
ax.barh(y, waits, color=colors, height=.54)
ax.set_yticks(y, labels)
ax.set_xscale("log")
ax.set_xlim(.1, 260)
ax.set_xticks([.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_xlabel("CPU reader wait, ms / update  (log scale)")
ax.set_title("B  Reader wait nearly disappears", loc="left", weight="bold", pad=14)
for pos, value in zip(y, waits, strict=True):
    ax.text(value * 1.12, pos, f"{value:.2f}", va="center", weight="bold")

ax = axes[1, 0]
for d, color, label in [(old, colors[0], "mmap / 2w × 2"), (new, colors[2], "pread / 8w × 4")]:
    steps = [row for row in d["steps"] if not row["warmup"]]
    rolling = 1 / np.convolve([row["seconds"] for row in steps], np.ones(32) / 32, mode="valid")
    ax.plot(np.arange(len(rolling)) + 32, rolling, color=color, label=label, lw=1.6)
ax.set_ylim(0, 20)
ax.set_xlim(0, 1008)
ax.set_title("C  Speed over the longer sequence", loc="left", weight="bold", pad=14)
ax.set_xlabel("Measured update  (32-update moving mean)")
ax.set_ylabel("Windows / second")
ax.legend(loc="lower left", frameon=False, fontsize=8.5)

ax = axes[1, 1]
cpu = [143.354992375, 10.109065375]
ax.barh([1, 0], cpu, color=[colors[0], colors[2]], height=.5)
ax.set_yticks([1, 0], ["mmap", "preadv"])
ax.set_xlim(0, 180)
ax.set_xlabel("Worker CPU ms / window, image preparation")
ax.set_title("D  Focused check of eight slow windows", loc="left", weight="bold", pad=14, size=10.5)
for pos, value in zip([1, 0], cpu, strict=True):
    ax.text(value + 3, pos, f"{value:.1f}", va="center", weight="bold")
ax.text(0, -.42, "Physical reads: 15.33 → 6.86 MB / window\nRequested JPEGs: 6.73 MB / window", transform=ax.transAxes,
        size=8.8, color=muted, va="top")
for i, axis in enumerate(axes.flat):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#cbd6dd")
    axis.grid(axis="y" if i == 2 else "x", color="#e5ebf0", linewidth=.7)
    axis.set_axisbelow(True)
fig.text(.055, .084, "Final reader wait: 0.27 ms mean; 0.41 ms P95; 0.40% of average step time.",
         size=11, weight="bold", color=colors[2])
fig.text(.055, .048, "Initial dual-hash verification: 270 s for 501 clips / 59.91 GB (shared within the final sweep; excluded from rates above).", size=8.8, color=muted)
fig.text(.055, .022, "Cache eviction is an OS hint. Single-machine diagnostics, not an accuracy claim. Source: captured JSON, 2026-10-09.", size=8.8, color=muted)
for suffix in ("png", "pdf", "svg"):
    p = ROOT / f"performance.{suffix}"
    fig.savefig(p, dpi=220)
    if suffix == "svg":
        p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
plt.close(fig)
