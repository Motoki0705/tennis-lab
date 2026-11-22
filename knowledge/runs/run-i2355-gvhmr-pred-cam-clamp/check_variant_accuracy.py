"""Compare static FP16 B=2 graph variants with the FP32 graph under ONNX Runtime.

CPU only.  The FP32-pinned variants change rounding, so they are compared with
the FP32 graph next to the unmodified FP16 graph rather than bit for bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_additive import eval_batch, feeds
from locked_inputs import checked_segments
from minimal_repro import run_onnx

FLOATS = ("pred_x", "pred_cam", "static_conf_logits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--fp32", required=True, type=Path)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    segments, _checkpoint = checked_segments(args.inputs, args.checkpoints)
    arrays = eval_batch(segments, 145, 2)
    reference = run_onnx(args.fp32, feeds(arrays, np.float32))
    report: dict[str, object] = {"batch": 2, "length": 145, "fp32": str(args.fp32)}
    for item in args.variant:
        label, path = item.split("=", 1)
        values = run_onnx(Path(path), feeds(arrays, np.float16))
        entry: dict[str, object] = {"graph": path}
        for name in FLOATS:
            if not np.isfinite(values[name]).all():
                raise RuntimeError(f"{label} {name} is not finite")
            entry[f"{name}_max_abs_vs_fp32"] = float(
                np.max(np.abs(values[name].astype(np.float64) - reference[name]))
            )
        for name in ("mask_out", "length_out"):
            np.testing.assert_array_equal(values[name], reference[name])
        report[label] = entry
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
