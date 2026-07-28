"""Probe the isolated NHT runtime without importing tennis dataset domains."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nht-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Publish runtime identity after importing CUDA and gsplat dependencies."""
    args = _parse_args()
    repository = args.nht_repository.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"NHT runtime probe refuses overwrite: {output}")
    import gsplat  # noqa: PLC0415

    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    payload = {
        "schema": "tennis_nht_runtime_probe_v1",
        "nht_commit": head,
        "torch_version": torch.__version__,
        "gsplat_module": str(Path(gsplat.__file__).resolve()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
