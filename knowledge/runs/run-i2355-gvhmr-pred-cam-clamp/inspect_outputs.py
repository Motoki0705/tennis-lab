"""Compare a GVHMR TensorRT plan's outputs with ONNX Runtime per batch and length.

For each requested shape, the same packed eval segments run through the plan
and through the plan's FP16 ONNX on CPU.  The report lists the length and
mask checks used by the #2305 probe, non-finite counts per output, and the
maximum difference from ONNX Runtime.  Nothing is accepted or rejected here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
for dependency in (
    HERE.parents[2],
    RUNS / "run-i2297-clipreid-tensorrt",
    RUNS / "run-i2305-gvhmr-tensorrt",
    HERE,
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from benchmark import write_json  # noqa: E402
from campaign_additive import feeds  # noqa: E402
from convert import require_queue  # noqa: E402
from locked_inputs import checked_segments  # noqa: E402
from minimal_repro import run_onnx  # noqa: E402
from model import PLAN, pack_sequences  # noqa: E402
from run_gvhmr import TensorRTSequenceRunner, collect_outputs, host_sample  # noqa: E402


def select(segments, length: int, batch: int, split: str | None):
    rows = [
        arrays
        for row, arrays in segments
        if row["length"] == length and (split is None or row["split"] == split)
    ][:batch]
    if len(rows) != batch:
        raise ValueError(f"Only {len(rows)} segments at L={length}")
    return pack_sequences(rows)


def describe(trt_values, onnx_values, length: int, batch: int) -> dict:
    entry: dict[str, object] = {
        "length_out": trt_values["length_out"].tolist(),
        "length_ok": trt_values["length_out"].tolist() == [length] * batch,
        "mask_all": bool(trt_values["mask_out"].all()),
        "mask_false_per_sequence": (~trt_values["mask_out"])
        .reshape(batch, -1)
        .sum(axis=1)
        .tolist(),
        "mask_equal_onnx": bool(
            np.array_equal(trt_values["mask_out"], onnx_values["mask_out"])
        ),
        "onnx_mask_all": bool(onnx_values["mask_out"].all()),
    }
    for name in ("pred_x", "pred_cam", "static_conf_logits"):
        value = trt_values[name].astype(np.float64)
        finite = np.isfinite(value)
        entry[name] = {
            "nonfinite": int((~finite).sum()),
            "onnx_nonfinite": int((~np.isfinite(onnx_values[name])).sum()),
            "max_abs_vs_onnx": float(
                np.max(np.abs(np.where(finite, value, 0) - onnx_values[name]))
            ),
        }
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--conversion", required=True, type=Path)
    parser.add_argument("--shape", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_queue()
    if args.output.exists():
        raise FileExistsError(args.output)
    conversion = json.loads((args.conversion / "conversion.json").read_text())
    graph = args.conversion / "export.onnx"
    segments, _checkpoint = checked_segments(args.inputs, args.checkpoints)
    report: dict[str, object] = {
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "conversion": str(args.conversion),
        "shapes": {},
    }
    for item in args.shape:
        split, batch_text, length_text = item.split(":")
        batch, length = int(batch_text), int(length_text)
        arrays = select(segments, length, batch, None if split == "any" else split)
        onnx_values = run_onnx(graph, feeds(arrays, np.float16))
        runner = TensorRTSequenceRunner(
            args.conversion / PLAN,
            length,
            batch_size=batch,
            input_profile=conversion["dynamic_profile"]["inputs"],
        )
        trt_values = collect_outputs(
            runner, [host_sample(runner, arrays)], length, batch_size=batch
        )
        report["shapes"][item] = describe(trt_values, onnx_values, length, batch)
        del runner
        torch.cuda.empty_cache()
        write_json(args.output, report)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
