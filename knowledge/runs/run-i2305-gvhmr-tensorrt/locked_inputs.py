"""Validate prepared GVHMR tensors without importing scene components."""

import hashlib
import json
from pathlib import Path

import numpy as np
from batching import batch_profile_points
from inputs import sha256_file
from model import CHECKPOINT, INPUT_WIDTHS, input_shapes

HERE = Path(__file__).resolve().parent


def sweep_profile(optimum, maximum):
    profile = json.loads((HERE / "shape-profile.json").read_text())
    del profile["batch"]
    batches = batch_profile_points(optimum, maximum)
    counts = profile["length_counts"]["eval"]
    if maximum > min(counts.values()):
        raise ValueError("GVHMR batch maximum exceeds an evaluation length bucket")
    shortest, longest = sorted(map(int, counts))
    lengths = (shortest, shortest, longest)
    bindings = [
        input_shapes(length, size)
        for size, length in zip(batches, lengths, strict=True)
    ]
    labels = ("min", "opt", "max")
    return {
        **profile,
        "profile_batches": dict(zip(labels, batches, strict=True)),
        "profile_lengths": dict(zip(labels, lengths, strict=True)),
        "inputs": {
            name: [list(bound[name]) for bound in bindings] for name in INPUT_WIDTHS
        },
        "outside_lengths": [shortest - 1, longest + 1],
    }


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
