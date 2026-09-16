"""Render one explicit B00 test sample to PNG in /tmp using the repo's float32 store decoder.

Source of truth: rgb.f32.npz (canonical float32 RGB). The companion rgb.png preview is
used only as a cross-check that the 8-bit conversion is faithful.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/kamimura/projects/tennis-lab")
from src.utils.data.float32_store import inspect_float32, read_float32  # noqa: E402

DATASET = Path(
    "/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B00/datasets/court"
)


def main() -> int:
    dataset = json.loads((DATASET / "dataset.json").read_text())
    test_samples = [s for s in dataset["samples"] if s["split"] == "test"]
    sample = test_samples[0]
    dtype, shape = inspect_float32(DATASET / sample["rgb"])
    rgb = read_float32(DATASET / sample["rgb"])
    if rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SystemExit(f"unexpected rgb array {rgb.dtype} {rgb.shape}")
    if (rgb.shape[1], rgb.shape[0]) != (sample["width"], sample["height"]):
        raise SystemExit("rgb resolution disagrees with dataset metadata")
    u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

    out = Path("/tmp/tcd_work/inputs/synth_court-sample-000897.png")
    Image.fromarray(u8, mode="RGB").save(out, compress_level=9)

    preview = np.asarray(Image.open(DATASET / sample["rgb_preview"]).convert("RGB"))
    diff = np.abs(preview.astype(np.int16) - u8.astype(np.int16))
    info = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "trajectory_group_id": sample["trajectory_group_id"],
        "view_id": sample["view_id"],
        "trajectory_frame_index": sample["trajectory_frame_index"],
        "dataset_index": sample["sample_index"],
        "rgb_store": sample["rgb"],
        "rgb_store_dtype": str(dtype),
        "rgb_store_shape": list(shape),
        "rgb_float_min": float(rgb.min()),
        "rgb_float_max": float(rgb.max()),
        "conversion": "uint8 = clip(round(float32 * 255), 0, 255)",
        "compare_to_rgb_preview_max_abs_diff": int(diff.max()),
        "compare_to_rgb_preview_mean_abs_diff": float(diff.mean()),
        "png_path": str(out),
        "png_bytes": out.stat().st_size,
        "png_pixels_sha256_rgb": hashlib.sha256(u8.tobytes()).hexdigest(),
        "projection_visible_point_count": sample["projection"]["visible_point_count"],
        "projection_coverage_modes": sample["projection"]["coverage_modes"],
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
