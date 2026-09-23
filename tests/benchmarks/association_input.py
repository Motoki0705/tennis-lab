"""CPU dataset benchmark; see the adjacent README for cross-revision comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor

import src.utils.hydra  # noqa: F401 -- registers the project path resolver
from src.tasks.blcs.data.association_datamodule import BLCSAssociationDataModule
from src.tasks.plcs.data.association_datamodule import PLCSAssociationDataModule
from src.utils.paths import PROJECT_ROOT


def _fingerprint(sample: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(sample.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(data_root: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    for task, module_type in (
        ("plcs", PLCSAssociationDataModule),
        ("blcs", BLCSAssociationDataModule),
    ):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        with initialize_config_dir(
            config_dir=str(PROJECT_ROOT / f"src/tasks/{task}/configs"),
            version_base="1.3",
        ):
            config = compose(
                config_name="train_association",
                overrides=[
                    f"paths.project_root={PROJECT_ROOT}",
                    f"paths.data_root={data_root}",
                    "run.gpus=0",
                    "data.num_workers=0",
                ],
            )
        dm = module_type(config)
        dm.setup("fit")
        assert dm.train_dataset is not None and dm.val_dataset is not None
        if len(dm.train_dataset) < 800 or len(dm.val_dataset) < 18:
            raise ValueError("Benchmark requires the production V2 train/val splits")
        dm.train_dataset[22]  # Same warmup and RNG consumption in both revisions.
        for stage, dataset, indices in (
            ("train", dm.train_dataset, (0, 1, 2, 17, 101, 255, 511, 799)),
            ("val", dm.val_dataset, (0, 1, 2, 17)),
        ):
            for index in indices:
                started = time.perf_counter()
                sample = dataset[index]
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "task": task,
                        "stage": stage,
                        "index": index,
                        "seconds": elapsed,
                        "shape": list(sample["object_uv"].shape),
                        "sample_sha256": _fingerprint(sample),
                    }
                )
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "data_root": str(data_root),
        "cpu_threads": 1,
        "seed": 42,
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
