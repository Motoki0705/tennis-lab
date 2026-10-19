"""Content identities and deterministic per-motion seeds for extraction runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from src.utils.io import load_json, save_json_atomic
from src.utils.seeding import seed_everything


def content_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_motion(seed: int, source_id: str, deterministic: bool) -> None:
    """Resume/subsetting cannot change a motion's random-number stream."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(int(content_digest([seed, source_id])[:8], 16))
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def publish_run(output_root: Path, description: dict[str, Any]) -> None:
    """Reject incompatible resumes before replacing any run metadata."""
    destination = output_root / "reproducibility.json"
    if destination.exists():
        previous = load_json(destination)
        if previous["sha256"] != description["sha256"]:
            raise RuntimeError(
                "Extraction inputs, settings, assets or code changed; select a new run.output_dir."
            )
    elif (output_root / "manifest.json").exists():
        raise RuntimeError(
            "Existing extraction has no reproducibility identity; select a new run.output_dir."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    save_json_atomic(description, destination)
    # JSON is a YAML subset; this atomic snapshot can be read by Hydra directly.
    save_json_atomic(description["config"], output_root / "config.yaml")
