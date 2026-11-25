"""Verify the two saved FastHMR smoke predictions and report numerical deltas."""

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

PINS = {
    "pt32": "217d6ee57df879eaa10ad156d835876252b9144bbfa84026bcc853a33a61ab81",
    "pt16": "16e2708a94f2d5170d9399f3d87e79619292c9afaa01ed904b89a77da3920264",
}
SOURCE_FRAMES = list(range(0, 16, 2))
FIELDS = {
    "vertices": (8, 6890, 3),
    "coco17": (8, 17, 3),
    "body_pose": (8, 69),
    "global_orient": (8, 3),
    "betas": (8, 10),
    "transl": (8, 3),
}


def read_one(root, mode):
    path = root / f"fasthmr-smoke-{mode}.npz"
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PINS[mode]:
            raise ValueError(f"Unexpected SHA-256 for {path}")
    with np.load(path, allow_pickle=False) as prediction:
        if prediction["source_frames"].tolist() != SOURCE_FRAMES:
            raise ValueError(f"Unexpected source timeline for {mode}")
        values = {name: prediction[name] for name in FIELDS}
    for name, shape in FIELDS.items():
        value = values[name]
        if (
            value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid {mode} {name}")
    if not (values["transl"][:, 2] > 0).all():
        raise ValueError(f"Unexpected {mode} camera translation")
    return values


def summarize(a, b):
    distance_mm = (
        np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1) * 1000
    )
    return {
        "count": distance_mm.size,
        "median_mm": float(np.median(distance_mm)),
        "p95_mm": float(np.quantile(distance_mm, 0.95)),
        "max_mm": float(distance_mm.max()),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    root = parser.parse_args().predictions
    fp32, fp16 = (read_one(root, mode) for mode in PINS)
    metrics = {
        key: summarize(fp32[key], fp16[key]) for key in ("coco17", "vertices", "transl")
    }
    print(
        json.dumps(
            {
                "source_frames": SOURCE_FRAMES,
                "prediction_sha256": PINS,
                "diagnostic": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
