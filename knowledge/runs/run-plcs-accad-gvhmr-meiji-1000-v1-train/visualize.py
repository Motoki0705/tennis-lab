"""CPU-only reproduction: python knowledge/runs/<run-id>/visualize.py.

Only bundled predictions and scene metadata are needed; no model or videos.
Legacy prediction writer zero-padded boolean masks across batches. Require a
nonzero target heading as well, and verify all excluded non-masked rows are zero.
Metrics here use float32 exports; official metrics used bf16 intermediates.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "scene_metadata.json").read_text())
    with np.load(root / "pred_test.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    heading_norm = np.linalg.norm(arrays["target_rotation"], axis=-1)
    extra_pad = ~arrays["padding_mask"] & (heading_norm == 0)
    for key, value in arrays.items():
        if key not in {"scene_ids", "padding_mask"}:
            assert np.all(value[extra_pad] == 0), key
    valid = ~arrays["padding_mask"] & ~extra_pad
    assert np.allclose(heading_norm[valid], 1, atol=1e-5)
    assert np.all(valid.sum(axis=1) > 0)
    ids = arrays["scene_ids"]
    sources = np.array([metadata["scenes"][s]["source"] for s in ids])
    scale = np.array(metadata["scale_xyz_m"])
    pred = arrays["pred_position"] * scale
    target = arrays["target_position"] * scale
    error = np.linalg.norm(pred - target, axis=-1)
    yaw = [
        np.arctan2(arrays[k][..., 1], arrays[k][..., 0])
        for k in ("pred_rotation", "target_rotation")
    ]
    angle = np.degrees(np.abs(np.angle(np.exp(1j * (yaw[0] - yaw[1])))))
    pose = np.linalg.norm(
        arrays["pred_canonical_pose"] - arrays["target_canonical_pose"], axis=-1
    ).mean(axis=-1)
    assert (
        abs(
            pose[valid].mean()
            - json.loads((root / "metrics.json").read_text())["canonical_mpjpe_m"]
        )
        < 1e-6
    )
    scene_error = np.array([e[m].mean() for e, m in zip(error, valid, strict=True)])
    summary = {
        "valid_frames": int(valid.sum()),
        "legacy_zero_padding_frames_excluded": int(extra_pad.sum()),
        "source_metrics": {},
    }
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for source, color in (("accad", "#1768ac"), ("gvhmr", "#d15b31")):
        mask = valid & (sources == source)[:, None]
        summary["source_metrics"][source] = {
            "scenes": int((sources == source).sum()),
            "frames": int(mask.sum()),
            "position_error_m": float(error[mask].mean()),
            "angular_error_deg": float(angle[mask].mean()),
            "canonical_mpjpe_m": float(pose[mask].mean()),
        }
        for ax, values, label in zip(
            axes,
            (error, angle, pose),
            ("3D root error (m)", "Heading error (deg)", "Canonical MPJPE (m)"),
            strict=True,
        ):
            x = np.sort(values[mask])
            ax.plot(
                x,
                np.arange(1, len(x) + 1) / len(x),
                color=color,
                label=f"{source.upper()} ({(sources == source).sum()} scenes)",
            )
            ax.set(xlabel=label, ylabel="Fraction of valid frames", ylim=(0, 1.01))
            ax.grid(alpha=0.2)
    axes[0].axvline(0.5, color="gray", ls=":")
    axes[1].axvline(15, color="gray", ls=":")
    axes[1].set_xlim(0, 180)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Mixed test split | 100 scenes | float32 export, frame-weighted")
    fig.savefig(root / "test_errors.png", dpi=150)
    plt.close(fig)
    selected = []
    for source in ("accad", "gvhmr"):
        indices = np.flatnonzero(sources == source)
        order = indices[np.argsort(scene_error[indices], kind="stable")]
        selected.append(
            (int(order[len(order) // 2]), f"{source.upper()} median-ranked")
        )
    selected.append((int(np.argmax(scene_error)), "Largest mean root error"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    summary["examples"] = []
    for column, (i, label) in enumerate(selected):
        m = valid[i]
        t = np.arange(m.sum()) / metadata["scenes"][ids[i]]["fps"]
        summary["examples"].append(
            {
                "scene_id": str(ids[i]),
                "selection": label,
                "position_error_m": float(scene_error[i]),
            }
        )
        ax = axes[0, column]
        for points, name, color in (
            (target, "Target", "#1768ac"),
            (pred, "Prediction", "#d15b31"),
        ):
            xy = points[i, m, :2]
            ax.plot(xy[:, 0], xy[:, 1], color=color, label=name, lw=1.7)
            ax.scatter(*xy[0], color=color, marker="o", s=30)
        ax.set(
            xlabel="Court X (m)",
            ylabel="Court Y (m)",
            title=f"{label}\n{ids[i]} | root error {scene_error[i]:.3f} m",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        ax = axes[1, column]
        for y, name, color in (
            (yaw[1], "Target", "#1768ac"),
            (yaw[0], "Prediction", "#d15b31"),
        ):
            ax.plot(t, np.degrees(y[i, m]), label=name, color=color)
        ax.set(
            xlabel="Time within saved test window (s)",
            ylabel="Heading (deg)",
            ylim=(-185, 185),
        )
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Held-out synthetic scenes: trajectories and headings (dot = start; yaw wraps at +/-180 deg)"
    )
    fig.savefig(root / "test_examples.png", dpi=150)
    plt.close(fig)
    (root / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
