"""Post-selection diagnostics only; does not change any selected candidate."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from probe import CLIPS, DatabaseConfig, generate, project, write_json

root = Path.cwd()
bundle = Path(__file__).parent
out = Path(sys.argv[1])
r = json.loads((out / "results.json").read_text())
cv2.setNumThreads(1)
rows = []
provenance = {}
panels = []
for v in r["views"]:
    config = v["database_config"]
    mid = {
        k: val if k == "slot_id" else [sum(val) / 2] * 2
        for k, val in config["camera"].items()
    }
    nominal = generate(
        DatabaseConfig.from_mapping({**config, "count": 1, "camera": mid})
    ).H[0]
    s = np.array(
        [
            [512 / 1920, 0, (512 / 1920 - 1) / 2],
            [0, 288 / 1080, (288 / 1080 - 1) / 2],
            [0, 0, 1.0],
        ]
    )
    baseline = np.asarray(v["baseline_global_H"])
    mismatch = np.linalg.norm(
        project(np.linalg.inv(s) @ nominal) - project(baseline), axis=1
    )
    R = np.array(v["approximate_v9_camera"]["R"])
    right = np.cross(R[2], [0, 0, 1])
    right /= np.linalg.norm(right)
    roll = float(np.degrees(np.arctan2(np.dot(R[1], right), np.dot(R[0], right))))
    rows.append(
        dict(
            view=v["view"],
            roll_relative_to_zero_roll_deg=roll,
            nominal_zero_roll_vs_v9_mean_px=float(mismatch.mean()),
            nominal_zero_roll_vs_v9_max_px=float(mismatch.max()),
        )
    )
    im = cv2.imread(str(out / v["view"] / "comparison.jpg"))
    small = cv2.resize(im, (320, 540))
    cv2.putText(
        small, v["view"], (4, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1
    )
    panels.append(small)
for video, clip in CLIPS:
    cache = (
        root
        / "outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005"
        / video
        / clip
    )
    source = (
        root
        / "data/tennis_multivew/processed/meiji_3cam/dataset/videos"
        / video
        / "clips"
        / clip
    )
    for path in [source / "clip.json", cache / "court.npz", cache / "court.json"]:
        provenance[str(path.relative_to(root))] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size=path.stat().st_size,
        )
    for path in sorted((source / "media").glob("*.mp4")):
        provenance[str(path.relative_to(root))] = dict(
            size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns
        )
    receipt = json.loads((cache / "court.json").read_text())
    provenance[f"{video}/{clip}/cache_identity"] = receipt.get("identity")
for path in [
    bundle / "probe.py",
    bundle / "repro.sh",
    bundle / "summarize.py",
    root / "src/synthetic_data_generation/court_calibration/database.py",
    root / "src/synthetic_data_generation/court_calibration/matching.py",
]:
    provenance[str(path.relative_to(root) if path.is_absolute() else path)] = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
write_json(bundle / "prior_diagnostics.json", rows)
write_json(bundle / "provenance.json", provenance)
cv2.imwrite(
    str(bundle / "contact_sheet.jpg"),
    np.concatenate(
        [np.concatenate(panels[i : i + 3], axis=1) for i in range(0, 9, 3)], axis=0
    ),
)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name in ["baseline", "initial", "refined"]:
    axes[0].plot(
        range(9),
        [v["stages"][name]["query"]["symmetric_mean_px"] for v in r["views"]],
        marker="o",
        label=name,
    )
    axes[1].plot(
        range(9),
        [v["stages"][name]["temporal"]["symmetric_mean_px"] for v in r["views"]],
        marker="o",
        label=name,
    )
    axes[2].plot(
        range(3),
        [v["heldout_manual"]["stages"][name]["mean_px"] for v in r["views"][:3]],
        marker="o",
        label=name,
    )
for ax, title, ylabel in zip(
    axes,
    ["Same RGB query", "Separate time RGB", "Held-out static manual layouts"],
    [
        "Truncated symmetric distance (512px image)",
        "Truncated symmetric distance (512px image)",
        "Mean keypoint error (1920px image)",
    ],
    strict=True,
):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("View index (clip-major, camera-minor)")
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(bundle / "metrics.png", dpi=140)
plt.close(fig)
summary = dict(
    views=9,
    ecc_estimates_returned=45,
    ecc_attempts=45,
    manual_independent_layouts=3,
    manual_points=42,
    manual_mean_px={
        name: float(
            np.mean(
                [v["heldout_manual"]["stages"][name]["mean_px"] for v in r["views"][:3]]
            )
        )
        for name in ["baseline", "initial", "refined"]
    },
    query_mean={
        name: float(
            np.mean(
                [v["stages"][name]["query"]["symmetric_mean_px"] for v in r["views"]]
            )
        )
        for name in ["baseline", "initial", "refined"]
    },
    temporal_mean={
        name: float(
            np.mean(
                [v["stages"][name]["temporal"]["symmetric_mean_px"] for v in r["views"]]
            )
        )
        for name in ["baseline", "initial", "refined"]
    },
)
write_json(bundle / "metrics.json", summary)
print(json.dumps(summary, indent=2))
print(json.dumps(rows, indent=2))
