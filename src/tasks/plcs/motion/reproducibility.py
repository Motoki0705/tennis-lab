"""Content identities and deterministic per-motion seeds for extraction runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.submodules.configuration import GvhmrDemoConfig
from src.tennis_scene.generate_dataset.manifest import file_sha256
from src.utils.io import load_json, save_json_atomic
from src.utils.paths import PROJECT_ROOT
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


def describe_run(
    *,
    config: dict[str, Any],
    model_runtime: GvhmrDemoConfig,
    selection: dict[str, object],
    dataset_root: Path,
) -> dict[str, Any]:
    """Hash actual loaded assets and implementation, not just their filenames."""
    code_root = PROJECT_ROOT
    assets = model_runtime.assets
    files = {
        "dino": assets.dino_checkpoint,
        "vitpose": assets.vitpose_checkpoint,
        "hmr2": assets.hmr2_checkpoint,
        "gvhmr": assets.gvhmr_checkpoint,
        **{f"bundled/{name}": path for name, path in asdict(assets.bundled).items()},
    }
    for directory, label in (
        (assets.body_models_dir, "body_models"),
        (assets.dino_repository, "dino_source"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {
                ".py",
                ".so",
                ".npz",
                ".npy",
                ".pkl",
                ".pt",
            }:
                files[f"{label}/{path.relative_to(directory)}"] = path
    hashes = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in files.items()
    }
    source_hashes = {
        str(path.relative_to(code_root)): file_sha256(path)
        for path in sorted((code_root / "src").rglob("*.py"))
    }
    versions = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
    }
    identity = {
        "schema_version": "plcs_extraction_repro_v1",
        "model_runtime": json.loads(json.dumps(asdict(model_runtime), default=str)),
        "selection": selection,
        "seed": config["run"]["seed"],
        "deterministic": config["run"]["deterministic"],
        "max_frames": config["run"]["max_frames"],
        "dataset_manifest_sha256": file_sha256(dataset_root / "dataset.json"),
        "assets": hashes,
        "source_sha256": content_digest(source_hashes),
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    untracked_sources = {
        name: (code_root / name).read_text()
        for name in untracked.stdout.splitlines()
        if Path(name).suffix in {".py", ".yaml", ".toml"}
    }
    return {
        "identity": identity,
        "sha256": content_digest(identity),
        "config": config,
        "source_files": source_hashes,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_diff": diff.stdout if diff.returncode == 0 else None,
        "untracked_sources": untracked_sources,
    }


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
