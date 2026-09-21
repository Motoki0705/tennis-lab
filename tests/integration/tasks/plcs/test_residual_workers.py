"""Exercise residual sampling in real spawn workers without a training loop.

The probe runs this file as a script so its top-level loader/dataset are
importable in spawned interpreters despite pytest's importlib module names.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, get_worker_info

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.data.augmentation.residual import fixed_six_camera_rig
from src.tasks.plcs.data.residual_datamodule import ResidualDataModule
from src.tasks.plcs.data.residual_dataset import ResidualDataset
from src.tasks.plcs.data.residual_types import CleanResidualScene

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _load_scene(path: Path) -> CleanResidualScene:
    joints = 17
    world: np.ndarray = np.zeros((48, joints, 3), dtype=np.float32)
    world[..., 0] = np.linspace(-0.5, 0.5, 48)[:, None]
    world[..., 0] += np.linspace(-0.2, 0.2, joints)[None]
    world[..., 1] = -6.0 + int(path.name.rsplit("_", 1)[1]) * 0.1
    world[..., 2] = np.linspace(0.7, 1.7, joints)[None]
    return CleanResidualScene(
        path.name, world, 30.0, fixed_six_camera_rig((1280, 720)), path.name
    )


class _WorkerDiagnosticsDataset(ResidualDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        worker = get_worker_info()
        sample["_worker_id"] = torch.tensor(-1 if worker is None else worker.id)
        sample["_pid"] = torch.tensor(os.getpid())
        sample["_opencv_threads"] = torch.tensor(cv2.getNumThreads())
        sample["_epoch"] = torch.tensor(self.epoch.value)
        sample["_spawn"] = torch.tensor(
            multiprocessing.get_start_method(allow_none=True) == "spawn"
        )
        return sample


def _batches_by_scene(loader: DataLoader[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {batch["scene_id"][0]: batch for batch in loader}


def _probe() -> dict[str, Any]:
    with initialize_config_dir(
        config_dir=str(REPOSITORY_ROOT / "src/tasks/plcs/configs"),
        version_base="1.3",
    ):
        overrides = [
            "data.batch_size=1",
            "data.sequence_length=12",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.min_views=2",
            "data.max_views=2",
            "data.cache_scenes=2",
            "augmentation.evaluation_views=2",
            "augmentation.error_mode=observation",
        ]
        config = validate_residual_config(
            compose(config_name="train_triangulation_residual", overrides=overrides)
        )
    paths = [Path(f"plcs_{index}") for index in range(4)]
    main = ResidualDataModule(config)
    main.datasets["train"] = _WorkerDiagnosticsDataset(
        paths, _load_scene, config, "train"
    )
    worker_config = replace(config, data=replace(config.data, num_workers=2))
    parallel = ResidualDataModule(worker_config)
    parallel.datasets["train"] = _WorkerDiagnosticsDataset(
        paths, _load_scene, worker_config, "train"
    )
    main_loader, worker_loader = main.train_dataloader(), parallel.train_dataloader()
    assert main_loader.multiprocessing_context is None
    assert not main_loader.persistent_workers
    assert worker_loader.multiprocessing_context.get_start_method() == "spawn"
    assert worker_loader.persistent_workers
    worker_loader.timeout = 30
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(2)
    first_epoch: dict[str, dict[str, Any]] | None = None
    worker_pids: set[int] | None = None
    try:
        for epoch in (0, 3):
            main.datasets["train"].set_epoch(epoch)
            parallel.datasets["train"].set_epoch(epoch)
            expected = _batches_by_scene(main_loader)
            actual = _batches_by_scene(worker_loader)
            assert set(expected) == set(actual) == {path.name for path in paths}
            current_pids = {int(batch["_pid"].item()) for batch in actual.values()}
            assert len(current_pids) == 2
            assert os.getpid() not in current_pids
            if worker_pids is not None:
                assert current_pids == worker_pids
            worker_pids = current_pids
            assert {int(batch["_worker_id"].item()) for batch in actual.values()} == {
                0,
                1,
            }
            for name in expected:
                assert expected[name].keys() == actual[name].keys()
                assert expected[name]["_opencv_threads"].item() == 2
                assert actual[name]["_opencv_threads"].item() == 1
                assert actual[name]["_spawn"].item()
                assert actual[name]["_epoch"].item() == epoch
                for key, value in expected[name].items():
                    if key == "scene_id":
                        assert value == actual[name][key]
                    elif not key.startswith("_"):
                        other = actual[name][key]
                        assert value.shape == other.shape and value.dtype == other.dtype
                        assert value.numpy().tobytes() == other.numpy().tobytes(), (
                            "plcs",
                            epoch,
                            name,
                            key,
                        )
            if first_epoch is not None:
                assert any(
                    not torch.equal(
                        first_epoch[name]["features"], actual[name]["features"]
                    )
                    for name in actual
                )
                assert any(
                    not torch.equal(
                        first_epoch[name]["target_world"], actual[name]["target_world"]
                    )
                    for name in actual
                )
            first_epoch = actual
    finally:
        cv2.setNumThreads(previous_threads)
        iterator = worker_loader._iterator
        if iterator is not None:
            workers = list(iterator._workers)
            iterator._shutdown_workers()
            worker_loader._iterator = None
            assert all(not worker.is_alive() for worker in workers)
    return {
        "task": "plcs",
        "samples_compared": 8,
        "persistent_worker_pids": sorted(worker_pids or []),
        "opencv_threads": 1,
        "epochs": [0, 3],
        "workers_stopped": True,
    }


def test_spawn_workers_match_single_process_and_observe_epoch_updates() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT)
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["task"] == "plcs"
    assert report["samples_compared"] == 8 and report["workers_stopped"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(_probe()))
