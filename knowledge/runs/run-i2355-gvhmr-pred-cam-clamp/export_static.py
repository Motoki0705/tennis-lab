"""Export one static-shape GVHMR denoiser ONNX with the exact pred_cam rewrite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
REPO = HERE.parents[2]
COMMON = RUNS / "run-i2297-clipreid-tensorrt"
GVHMR = RUNS / "run-i2305-gvhmr-tensorrt"
for dependency in (REPO, COMMON, GVHMR, HERE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from benchmark import write_json  # noqa: E402
from convert_gvhmr import export_network  # noqa: E402
from inputs import sha256_file  # noqa: E402
from locked_inputs import checked_segments  # noqa: E402
from model import load_network, pack_sequences  # noqa: E402
from rewrite_pred_cam import rewrite_pred_cam_clamp  # noqa: E402

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def export_static(
    inputs: Path,
    checkpoints: Path,
    output: Path,
    *,
    batch: int,
    length: int,
    dtype: str,
) -> dict[str, object]:
    segments, checkpoint = checked_segments(inputs, checkpoints)
    selected = [
        arrays
        for row, arrays in segments
        if row["split"] == "eval" and row["length"] == length
    ][:batch]
    if len(selected) != batch:
        raise ValueError(f"Only {len(selected)} eval segments have L={length}")
    arrays = pack_sequences(selected)
    output.mkdir(parents=True)
    source = output / "scatter.onnx"
    target = output / "concat.onnx"
    network = load_network(checkpoint).to(dtype=DTYPES[dtype])
    operations = export_network(
        network, {length: arrays}, source, DTYPES[dtype], length, dynamic_length=False
    )
    rewrite = rewrite_pred_cam_clamp(source, target)
    report = {
        "batch": batch,
        "length": length,
        "dtype": dtype,
        "operations": len(operations),
        "rewrite": rewrite,
        "concat_sha256": sha256_file(target),
    }
    write_json(output / "export.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    args = parser.parse_args()
    report = export_static(
        args.inputs.resolve(strict=True),
        args.checkpoints,
        args.output.resolve(),
        batch=args.batch,
        length=args.length,
        dtype=args.dtype,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
