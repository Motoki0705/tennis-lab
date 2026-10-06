"""NPZ file boundary around the public checkpoint predictor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor


def predict_file(
    checkpoint: Path,
    source: Path,
    output: Path,
    *,
    device: str,
    seed: int,
    batch_size: int,
) -> None:
    if output.exists() or output.resolve() in (checkpoint.resolve(), source.resolve()):
        raise FileExistsError(
            "Prediction output must be a new file separate from its inputs"
        )
    predictor = RefinerPredictor.from_checkpoint(checkpoint, device=device)
    metadata = predictor.metadata
    dimensions = predictor.model.config.dimensions
    with np.load(source, allow_pickle=False) as values:
        required = {"coordinates", "missing", "fps"}
        if set(values.files) != required:
            raise ValueError(f"Input NPZ must contain exactly {sorted(required)}")
        coordinates = np.asarray(values["coordinates"], dtype=np.float32)
        missing = values["missing"]
        fps = float(values["fps"])
    if not np.isclose(fps, metadata["fps"], rtol=0, atol=1e-6):
        raise ValueError(
            f"Input FPS {fps} does not match checkpoint FPS {metadata['fps']}; resample explicitly"
        )
    if (
        coordinates.ndim != 3
        or coordinates.shape[-1] != dimensions
        or missing.dtype != np.bool_
    ):
        raise ValueError("Require coordinates (B,T,D) and a boolean missing mask")
    with torch.no_grad():
        predicted = predictor.predict(
            torch.from_numpy(coordinates),
            torch.from_numpy(missing),
            fps=fps,
            seed=seed,
            batch_size=batch_size,
        )
    result = predicted.coordinates.cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            coordinates=result,
            event_probability=predicted.event_probability.cpu().numpy(),
            input_missing=missing,
            fps=np.asarray(fps),
            checkpoint_sha256=np.asarray(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ),
            schema=np.asarray("ball_refiner_3d.prediction.v1"),
        )
    temporary.replace(output)
