"""Refine an offline coordinate+mask NPZ using an explicit trained checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.coordinates.inference import (
    load_checkpoint,
    refine_coordinates,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.device import resolve_device

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.predict_coordinates",
    fields=(
        BoundaryPathField("checkpoint", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("input", PathRole.DATA, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
    ),
)


def predict_file(checkpoint: Path, source: Path, output: Path, *, device: str, seed: int, batch_size: int) -> None:
    if output.exists() or output.resolve() in (checkpoint.resolve(), source.resolve()):
        raise FileExistsError("Prediction output must be a new file separate from its inputs")
    resolved_device = resolve_device(device)
    model, metadata = load_checkpoint(checkpoint, resolved_device)
    dimensions = model.config.dimensions
    with np.load(source, allow_pickle=False) as values:
        required = {"coordinates", "missing", "fps"} | ({"image_size_wh"} if dimensions == 2 else set())
        if set(values.files) != required:
            raise ValueError(f"Input NPZ must contain exactly {sorted(required)}")
        coordinates = np.asarray(values["coordinates"], dtype=np.float32)
        missing = values["missing"]
        fps = float(values["fps"])
        image_size = values["image_size_wh"] if dimensions == 2 else None
    if not np.isclose(fps, metadata["fps"], rtol=0, atol=1e-6):
        raise ValueError(f"Input FPS {fps} does not match checkpoint FPS {metadata['fps']}; resample explicitly")
    if coordinates.ndim != 3 or coordinates.shape[-1] != dimensions or missing.dtype != np.bool_:
        raise ValueError("Require coordinates (B,T,D) and a boolean missing mask")
    scale: NDArray[np.float32] = np.ones(dimensions, np.float32)
    if image_size is not None:
        if image_size.shape != (2,) or not np.isfinite(image_size).all() or (image_size < 2).any():
            raise ValueError("image_size_wh must be a valid source-pixel grid")
        scale = np.asarray((image_size - 1) / np.array([1279, 719]), np.float32)
    with torch.no_grad():
        predicted = refine_coordinates(model, torch.from_numpy(coordinates / scale).to(resolved_device),
                                       torch.from_numpy(missing).to(resolved_device), seed=seed, batch_size=batch_size)
    result = predicted.cpu().numpy() * scale
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, coordinates=result, input_missing=missing, fps=np.asarray(fps),
                            checkpoint_sha256=np.asarray(hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
                            schema=np.asarray("ball_refiner.coordinate_prediction.v1"))
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.checkpoint, args.input, args.output)):
        parser.error("checkpoint, input and output must be explicit absolute paths")
    roots = RuntimePathRoots(project_root=Path.cwd(), data_root=args.input.parent, checkpoint_root=args.checkpoint.parent,
                             output_root=args.output.parent, artifact_root=args.output.parent, cache_root=args.output.parent,
                             external_asset_root=args.output.parent)
    paths = PATH_BOUNDARY.validate({"checkpoint": args.checkpoint, "input": args.input, "output": args.output}, resolver=PathResolver(roots))
    torch.set_num_threads(2)
    predict_file(paths.declared("checkpoint").path, paths.declared("input").path, paths.declared("output").path,
                 device=args.device, seed=args.seed, batch_size=args.batch_size)
    print(str(args.output))


if __name__ == "__main__":
    main()
