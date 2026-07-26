"""Content identity for reproducible synthetic-data publishers."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

from src.synthetic_data_generation.provider.bundle import sha256_file

FULL_SCALE_RELEVANT_FILES = (
    "src/synthetic_data_generation/rendering/cpu_fake_renderer.py",
    "src/synthetic_data_generation/rendering/renderer_port.py",
    "src/synthetic_data_generation/dataset/full_scale_dataset.py",
    "src/synthetic_data_generation/scripts/publish_b00_full_scale_dataset.py",
    "src/synthetic_data_generation/configs/publish_b00_full_scale_dataset.yaml",
)


def compute_code_identity(
    repo_root: Path,
    relevant_files: Sequence[str] = FULL_SCALE_RELEVANT_FILES,
) -> str:
    """Hash the revision plus committed and working-tree forms of relevant files."""
    digest = hashlib.sha256()
    digest.update(_git(repo_root, "rev-parse", "HEAD").encode())
    digest.update(
        subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *relevant_files],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    )
    for relative in relevant_files:
        path = repo_root / relative
        digest.update(relative.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
