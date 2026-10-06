"""Standalone qualitative plots from saved inference arrays."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predictions(arrays: dict[str, np.ndarray], output: Path, *, dimensions: int) -> None:
    rallies = np.unique(arrays["rally_id"])[:3]
    fig, axes = plt.subplots(len(rallies), dimensions, figsize=(6 * dimensions, 3 * len(rallies)), squeeze=False)
    for row, rally in enumerate(rallies):
        selected = (arrays["rally_id"] == rally) & (arrays["view_id"] == 0)
        frames = arrays["frame_id"][selected]
        missing = arrays["missing"][selected]
        for axis in range(dimensions):
            ax = axes[row, axis]
            ax.plot(frames, arrays["target"][selected, axis], label="BLCS truth", color="#167d58", linewidth=1.4)
            ax.scatter(frames[~missing], arrays["input"][selected, axis][~missing], label="Noisy observations", color="#999999", s=4, alpha=0.5)
            ax.plot(frames, arrays["prediction"][selected, axis], label="Refiner", color="#d84b2b", linewidth=1.0)
            ax.fill_between(frames, 0, 1, where=missing, transform=ax.get_xaxis_transform(), color="#478ecc", alpha=0.15)
            ax.set_title(f"Rally {rally}, {'XYZ'[axis]} (m)")
            ax.set_xlabel("Frame")
            if row == 0 and axis == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)
