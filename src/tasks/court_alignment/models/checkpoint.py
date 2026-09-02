"""Strict model-only checkpoint loading for court alignment.

Historical court-alignment checkpoints predate the repository-wide court
coordinate metadata contract.  Their model payload is nevertheless explicit:
every tensor under ``state_dict`` is namespaced by ``model.``.  This module
owns that narrow compatibility boundary without weakening shared task
checkpoint validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn


def _extract_model_state(
    checkpoint: object, *, checkpoint_path: Path
) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Court-alignment checkpoint {checkpoint_path} root must be a mapping."
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Court-alignment checkpoint {checkpoint_path} 'state_dict' must be a mapping."
        )
    if not state:
        raise ValueError(
            f"Court-alignment checkpoint {checkpoint_path} 'state_dict' must not be empty."
        )
    if any(not isinstance(key, str) for key in state):
        raise TypeError(
            f"Court-alignment checkpoint {checkpoint_path} state_dict keys must be strings."
        )
    invalid_keys = sorted(
        cast(str, key) for key in state if not cast(str, key).startswith("model.")
    )
    if invalid_keys:
        raise ValueError(
            f"Court-alignment checkpoint {checkpoint_path} state_dict contains "
            "mixed or invalid prefixes; every key must start with 'model.': "
            f"{invalid_keys[:5]}."
        )

    stripped: dict[str, Tensor] = {}
    for raw_key, raw_value in state.items():
        source_key = cast(str, raw_key)
        key = source_key[len("model.") :]
        if not key:
            raise ValueError(
                f"Court-alignment checkpoint {checkpoint_path} contains an empty "
                "key after removing 'model.'."
            )
        if not isinstance(raw_value, Tensor):
            raise TypeError(
                f"Court-alignment checkpoint {checkpoint_path} value "
                f"{source_key!r} must be a tensor."
            )
        if key in stripped:
            raise ValueError(
                f"Court-alignment checkpoint {checkpoint_path} contains duplicate "
                f"model key {key!r} after prefix removal."
            )
        stripped[key] = raw_value
    return stripped


def _validate_exact_model_state(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    checkpoint_path: Path,
) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    shape_mismatches = sorted(
        (
            key,
            tuple(state_dict[key].shape),
            tuple(expected[key].shape),
        )
        for key in set(expected) & set(state_dict)
        if state_dict[key].shape != expected[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Court-alignment checkpoint {checkpoint_path} does not exactly match "
            f"the model: missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={shape_mismatches[:5]}."
        )


def load_court_alignment_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Load only a strict ``model.`` state into ``model`` on CPU.

    Optimizer, scheduler, loop, epoch, and every other root-level checkpoint
    field are intentionally ignored.  Path-role containment is validated by
    the caller that owns the runtime boundary.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("Court-alignment checkpoint target must be a torch module.")
    if not isinstance(checkpoint_path, Path) or not checkpoint_path.is_absolute():
        raise ValueError("Court-alignment checkpoint_path must be an absolute Path.")
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError(
            f"Court-alignment checkpoint must be an ordinary file: {checkpoint_path}"
        )
    checkpoint: object = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = _extract_model_state(
        checkpoint,
        checkpoint_path=checkpoint_path,
    )
    _validate_exact_model_state(
        model,
        state_dict,
        checkpoint_path=checkpoint_path,
    )
    # strict=True is retained even after the explicit diagnostics above so the
    # PyTorch loader remains the final architecture compatibility authority.
    model.load_state_dict(state_dict, strict=True)

    metadata: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "state_dict_key_count": len(state_dict),
    }
    if isinstance(checkpoint, Mapping):
        for key in ("epoch", "global_step", "pytorch-lightning_version"):
            value = checkpoint.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                metadata[key] = value
    return metadata


__all__ = ["load_court_alignment_model_checkpoint"]
