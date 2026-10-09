"""CPU-only failure receipts, including when the CUDA context is unusable."""
from __future__ import annotations

import json
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def batch_identity(batch: dict[str, Any]) -> dict[str, Any]:
    identity = {}
    for key in ("clip_id", "start", "frame_step", "frame_indices"):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Failure metadata must remain on CPU")
            value = value.tolist()
        identity[key] = value
    return identity


@contextmanager
def record_failure(output: Path, context: dict[str, Any]) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        # Do not inspect CUDA values, synchronize, or save a broken checkpoint.
        receipt = dict(context, exception_type=type(error).__name__, message=str(error),
                       traceback="".join(traceback.format_exception(error)),
                       caveat="surface_phase is where an error surfaced, not proof of its origin")
        try:
            (output / "failure.json").write_text(json.dumps(receipt, indent=2))
        except OSError as receipt_error:
            error.add_note(f"Could not save failure receipt: {receipt_error}")
        raise
