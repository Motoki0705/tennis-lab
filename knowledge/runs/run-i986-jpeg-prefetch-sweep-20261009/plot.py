"""Captured short sweep; keep the later failed long trial visible."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
labels = ["Sync / 2 workers / BS1", "Async / 2 workers / BS1", "Async / 4 workers / BS1",
          "Async / 2 workers / BS2", "RAM JPEG / async / BS1"]
names = ["sync-w2-b1", "async-w2-b1", "async-w4-b1", "async-w2-b2", "async-prepared-b1"]
data = [json.loads((ROOT / f"{name}.json").read_text()) for name in names]
ink, muted, teal, red = "#182c3d", "#526777", "#008b80", "#b9473f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink,
                     "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
fig = plt.figure(figsize=(12, 8.5), facecolor="white")
grid = fig.add_gridspec(2, 2, left=.26, right=.94, top=.78, bottom=.18, hspace=.75, wspace=.35,
                       height_ratios=[1.15, 1])
fig.text(.06, .94, "Input pipeline: short trials are not enough", size=21, weight="bold")
fig.text(.06, .89, "RTX 5060 Ti 16 GB  |  BF16  |  Conv2d + query-only  |  32-frame windows", color=muted)
fig.text(.06, .85, "Matched sweep: 204 windows; first 12 excluded. Initial verification: 308 s for 22.33 GB.",
         color=muted, size=10)
ax = fig.add_subplot(grid[0, :])
y = np.arange(5)[::-1]
values = [d["windows_per_second"] for d in data]
colors = ["#8292a4", teal, "#6b9b93", "#8da9a5", "#d9e7e5"]
ax.barh(y, values, height=.58, color=colors)
ax.set_yticks(y, labels)
ax.set_xlim(0, 18)
ax.set_xlabel("Windows / second  (higher is faster)", labelpad=8)
ax.set_title("A  Same windows, different input scheduling", loc="left", weight="bold", pad=13)
for pos, value in zip(y, values, strict=True):
    ax.text(value + .16, pos, f"{value:.2f}", va="center", weight="bold")
ax.axvline(values[-1], color="#74978e", ls="--", lw=1, zorder=0)
ax.text(.99, -.28, "RAM reference includes JPEG decode", transform=ax.transAxes,
        color=muted, ha="right", size=8.5)

ax2 = fig.add_subplot(grid[1, 0])
long = json.loads((ROOT.parent / "run-i986-nvjpeg-long-reader-20261009/report.json").read_text())
ax2.bar([0, 1], [values[1], long["windows_per_second"]], color=[teal, red], width=.58)
ax2.set_xticks([0, 1], ["204 windows", "1,020 windows"])
ax2.set_ylim(0, 18)
ax2.set_ylabel("Windows / second")
ax2.set_title("B  Longer trial exposed a regression", loc="left", weight="bold", pad=13, size=10)
for x, value in enumerate([values[1], long["windows_per_second"]]):
    ax2.text(x, value + .4, f"{value:.2f}", ha="center", weight="bold")
ax2.text(.5, -.30, "Both: async / 2 workers / BS1 / mmap reader", transform=ax2.transAxes,
         ha="center", size=8.3, color=muted)

ax3 = fig.add_subplot(grid[1, 1])
waiting = long["mean_loader_wait_seconds"] * 1000
other = long["mean_step_seconds"] * 1000 - waiting
ax3.bar(0, waiting, color=red, width=.52, label="Input-ready wait")
ax3.bar(0, other, bottom=waiting, color="#547589", width=.52, label="Remaining step")
ax3.set_xticks([0], ["1,020-window trial"])
ax3.set_ylabel("Milliseconds / update")
ax3.set_ylim(0, 205)
ax3.set_xlim(-.7, .7)
ax3.set_title("C  The long run remained input-bound", loc="left", weight="bold", pad=13, size=10)
ax3.text(0, waiting / 2, f"{waiting:.0f} ms", ha="center", va="center", weight="bold", color="white")
ax3.text(0, waiting + other / 2, f"{other:.0f} ms", ha="center", va="center", color="white")
ax3.legend(loc="upper center", bbox_to_anchor=(.5, -.21), frameon=False, ncol=2, fontsize=8)
for axis in (ax, ax2, ax3):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#cbd6dd")
    axis.grid(axis="x" if axis is ax else "y", color="#e6ecf0", linewidth=.6)
    axis.set_axisbelow(True)
fig.text(.06, .075, "Conclusion: do not adopt from the short sweep alone. Investigate first-read page faults and exact-range I/O.",
         size=10, weight="bold")
fig.text(.06, .035, "One trial per condition. Verification/setup costs excluded above; no accuracy or convergence claim. Source: captured JSON, 2026-10-09.",
         size=8.5, color=muted)
for suffix in ("png", "pdf", "svg"):
    path = ROOT / f"performance.{suffix}"
    fig.savefig(path, dpi=210)
    if suffix == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
