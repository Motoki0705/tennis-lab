"""Render the captured native RGB benchmark; CPU only, no rerun of training."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DATA = {name: json.loads((ROOT / f"{name}.json").read_text())
        for name in ("compute-off", "compute-default", "pipeline-off", "pipeline-default")}
COLORS = ["#70839a", "#008b80"]
INK, MUTED = "#162b40", "#506277"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": INK,
                     "axes.edgecolor": "#cfdae2", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})
fig, axs = plt.subplots(2, 2, figsize=(12, 8.6))
fig.subplots_adjust(left=.10, right=.94, top=.79, bottom=.22, wspace=.42, hspace=.82)
fig.text(.06, .944, "Native RGB + compiled MDD detector", size=21, weight="bold")
fig.text(.06, .900, "RTX 5060 Ti 16 GB  |  BF16  |  32 frames at 720p  |  batch size 1", size=11, color=MUTED)
fig.text(.06, .865, "Same RGB-input model in both conditions; fixed FP32 MDD is inside forward.", size=10, color=MUTED)


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", pad=17, size=12, weight="bold")
    ax.set_xlabel(xlabel, labelpad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color="#e5ebf0", linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-.60, 1.60)
    ax.set_yticks([1, 0], ["Eager", "Compiled"])
    ax.xaxis.set_major_locator(MaxNLocator(5))


for ax, mode, title, xmax, note in [
    (axs[0, 0], "compute", "A   GPU-resident training", 25,
     "24 measured updates; decode and transfer excluded"),
    (axs[0, 1], "pipeline", "B   With CPU reader + transfer", 2.1,
     "Same 96 measured windows; 8 loader workers"),
]:
    vals = [DATA[f"{mode}-{name}"]["windows_per_second"] for name in ("off", "default")]
    style(ax, title, "Windows / second   (higher is faster)")
    ax.barh([1, 0], vals, height=.49, color=COLORS)
    ax.set_xlim(0, xmax)
    for y, val in zip([1, 0], vals, strict=True):
        ax.text(val + .02 * xmax, y, f"{val:.2f}", va="center", weight="bold", size=12)
    ax.text(.98, .99, f"{vals[1] / vals[0]:.2f}×", ha="right", va="top",
            transform=ax.transAxes, color=COLORS[1], size=15, weight="bold")
    ax.text(0, -.38, note, transform=ax.transAxes, size=8.5, color=MUTED)

ax = axs[1, 0]
style(ax, "C   Wall-clock step with reader", "Mean seconds / update   (lower is faster)")
waits = [DATA[f"pipeline-{name}"]["mean_loader_wait_seconds"] for name in ("off", "default")]
totals = [DATA[f"pipeline-{name}"]["mean_step_seconds"] for name in ("off", "default")]
rest = [total - wait for total, wait in zip(totals, waits, strict=True)]
ax.barh([1, 0], waits, height=.49, color="#d09b4a", label="Waiting for next batch")
ax.barh([1, 0], rest, left=waits, height=.49, color="#36566d", label="Transfer + training + overhead")
for y, wait, total in zip([1, 0], waits, totals, strict=True):
    ax.text(wait / 2, y, f"{wait:.3f}s", ha="center", va="center", size=10, color=INK)
    ax.text(total + .02, y, f"{total:.3f}s", va="center", weight="bold", size=10)
ax.set_xlim(0, 1.12)
ax.legend(loc="upper left", bbox_to_anchor=(0, -.35), frameon=False, fontsize=8.5,
          handlelength=1.3, borderaxespad=0)

ax = axs[1, 1]
style(ax, "D   GPU-resident peak reserved VRAM", "GiB   (PyTorch allocator only)")
vals = [DATA[f"compute-{name}"]["peak_reserved_bytes"] / 2**30 for name in ("off", "default")]
ax.barh([1, 0], vals, height=.49, color=COLORS)
ax.set_xlim(0, 3.1)
for y, val in zip([1, 0], vals, strict=True):
    ax.text(val + .06, y, f"{val:.2f}", va="center", weight="bold", size=12)
ax.text(0, -.38, "CUDA context / display memory excluded", transform=ax.transAxes, size=8.5, color=MUTED)

fig.text(.06, .055, "One short trial per condition; warmup excluded. Panel A/B scales differ. No accuracy claim.",
         size=9, color=MUTED)
fig.text(.06, .025, "Source: captured JSON in run-i986-native-rgb-compile-20261009 | PyTorch 2.13.0+cu130 | Inductor default", size=8, color=MUTED)
for suffix in ("png", "svg", "pdf"):
    fig.savefig(ROOT / f"performance.{suffix}", dpi=220)
plt.close(fig)
