"""Inspect alpha in a projected net band; the band is not a semantic net mask."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = Path(__file__).parent
    camera = json.loads((run / "camera-request.json").read_text())["cameras"][0]
    alignment = json.loads(args.alignment.read_text())
    court = next(
        c
        for c in alignment["layout"]["courts"]
        if c["court_instance_id"] == "court-000"
    )
    adapter = alignment["metric_scene_adapter"]
    scale = adapter["nht_scene_units_per_metre"]
    transform = np.asarray(adapter["nht_scene_from_metric_scene"]).reshape(
        4, 4
    ) @ np.asarray(court["scene_from_court"]).reshape(4, 4)
    view = np.linalg.inv(np.asarray(camera["camera_to_scene"]))
    points = np.asarray(
        [
            [-2, 0, 0.7, 1],
            [2, 0, 0.7, 1],
            [-2, 0, 1.1, 1],
            [2, 0, 1.1, 1],
            [0, 0, 0.914, 1],
        ]
    )
    local = (view @ transform @ points.T).T[:, :3]
    k = np.asarray(camera["intrinsics"]["matrix"])
    uv_h = local @ k.T
    uv = uv_h[:, :2] / uv_h[:, 2:]
    u0, v0 = np.floor(uv[:4].min(axis=0)).astype(int)
    u1, v1 = np.ceil(uv[:4].max(axis=0)).astype(int)
    if not (0 <= u0 < u1 < 256 and 0 <= v0 < v1 < 144):
        raise ValueError("Projected net band is not inside the probe camera")
    j, i = np.floor(uv[4]).astype(int)
    full = np.load(args.arrays / "eye-0-full-alpha.npy", allow_pickle=False)[..., 0]
    output = {
        "schema_version": 1,
        "camera": "court-eye-0",
        "court": "court-000",
        "roi_is_semantic_mask": False,
        "box_uv_inclusive_exclusive": [int(u0), int(v0), int(u1), int(v1)],
        "court_net_center_uv": uv[4].tolist(),
        "net_center_camera_z_m": float(local[4, 2] / scale),
        "nearest_sample_index_ij": [int(i), int(j)],
        "full_alpha_roi_mean": float(full[v0:v1, u0:u1].mean()),
        "full_alpha_at_sample": float(full[i, j]),
        "prefix": [],
    }
    for cutoff in (6, 10):
        with np.load(
            args.arrays / f"eye-0-prefix-{cutoff}m.npz", allow_pickle=False
        ) as archive:
            alpha = archive["alpha"][..., 0]
        roi = alpha[v0:v1, u0:u1]
        output["prefix"].append(
            {
                "cutoff_m": cutoff,
                "roi_alpha_mean": float(roi.mean()),
                "roi_mid_alpha_pixels": int(((roi > 0.02) & (roi < 0.9)).sum()),
                "roi_pixel_count": int(roi.size),
                "sample_alpha": float(alpha[i, j]),
                "sample_transmittance": float(1 - alpha[i, j]),
            }
        )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
