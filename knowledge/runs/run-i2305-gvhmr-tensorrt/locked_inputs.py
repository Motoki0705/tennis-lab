"""Validate prepared GVHMR tensors without importing scene components."""

import hashlib
import json
from pathlib import Path

import numpy as np
from inputs import sha256_file
from model import CHECKPOINT, INPUT_WIDTHS, input_shapes

HERE = Path(__file__).resolve().parent


def tensor_addresses(arrays):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for key, value in sorted(arrays.items())
    }


def load_segments(directory, lock):
    """Return manifest-ordered tensors after verifying every content digest."""
    for row in lock["segments"]:
        path = directory / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as bundle:
            arrays = dict(bundle)
        if tensor_addresses(arrays) != row["tensors"]:
            raise ValueError(f"Tensor lock mismatch: {path}")
        length = row["length"]
        actual = {key: arrays[key].shape for key in INPUT_WIDTHS}
        if actual != input_shapes(length) or arrays["length"].tolist() != [length]:
            raise ValueError(f"Inconsistent network input length: {path}")
        if not arrays["sequence_mask"].all():
            raise ValueError(f"Unexpected padding in a selected segment: {path}")
        yield row, arrays


def checked_segments(directory, checkpoints):
    lock = json.loads((HERE / "tensor-lock.json").read_text())
    if json.loads((directory / "tensor-lock.json").read_text()) != lock:
        raise ValueError("Prepared input receipt differs from the tracked tensor lock")
    checkpoint = checkpoints / CHECKPOINT
    if sha256_file(checkpoint) != lock["assets"]["gvhmr"]["sha256"]:
        raise ValueError("GVHMR checkpoint digest differs from the tensor lock")
    return list(load_segments(directory, lock)), checkpoint
