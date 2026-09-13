"""Bounded research comparison using real court data, model/loss and NHT CLI."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.rendering.nht.contracts import NHTRenderCamera
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.court_detection.data.collate import court_detection_collate
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.data.float32_store import read_float32, write_float32


def transfer(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: transfer(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(transfer(item) for item in value)
    return value


def response(process: subprocess.Popen[str]) -> dict[str, Any]:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Renderer exited: {process.poll()}")
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            print("NHT:", line.rstrip(), flush=True)
            continue
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--nht-worker", type=Path, required=True)
    parser.add_argument("--nht-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--targets", choices=["kp", "all"], default="all")
    parser.add_argument(
        "--geometry", choices=["evaluation", "training"], default="evaluation"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = Path(__file__).resolve().parents[2]
    overrides = [
        "data/source=synthetic_court",
        f"data/processing={args.targets}",
        "data/augmentation=pose_safe",
        "training=lora",
        "data.num_workers=0",
        "training.compile.enabled=false",
        f"paths.data_root={args.repo / 'data'}",
        f"paths.checkpoint_root={args.repo / 'outputs'}",
        f"paths.external_asset_root={args.repo / 'third_party'}",
        f"paths.cache_root={args.repo / '.cache'}",
    ]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/court_detection/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=overrides)
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, args.output / "config.yaml")
    dm = CourtDetectionDataModule(cfg)
    pipeline = (
        dm._eval_pipeline if args.geometry == "evaluation" else dm._train_pipeline
    )
    records = pipeline.input_layer.records("train")[:64]
    scene_root = args.repo / "data/synthetic_data_generation/scenes/B00"
    dataset = json.loads((scene_root / "datasets/court/dataset.json").read_text())
    by_id = {s["sample_id"]: s for s in dataset["samples"]}
    alignment = load_alignment_result(scene_root / "alignment/alignment.json")
    cameras = []
    for record in records:
        sample = by_id[record.payload["source_sample_id"]]
        camera = SceneCamera.from_dict(sample["camera"])
        cameras.append(
            NHTRenderCamera(
                camera_id=camera.camera_id,
                width=camera.width,
                height=camera.height,
                intrinsics=camera.intrinsics,
                camera_to_scene=alignment.metric_adapter.nht_from_metric_camera(
                    camera.camera_to_scene
                ),
            ).to_dict()
        )
    request = args.output / "cameras.json"
    request.write_text(
        json.dumps({"schema": "nht_render_request_v1", "cameras": cameras})
    )
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "batch_size": 8,
        "steps": args.steps,
        "targets": args.targets,
        "compile": False,
        "training": "lora",
        "scenes": {},
    }

    def save() -> None:
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(
        prefix="court-ondemand-", dir="/dev/shm"
    ) as temporary:
        buffer = Path(temporary) / "batch.npy"
        # Public unmodified CLI: include Python startup, export validation, file writes.
        for count in [1, 8]:
            subrequest = args.output / f"cold-{count}.json"
            subrequest.write_text(
                json.dumps(
                    {"schema": "nht_render_request_v1", "cameras": cameras[:count]}
                )
            )
            out = Path(temporary) / f"cold-{count}"
            start = time.perf_counter()
            subprocess.run(
                [
                    "nht-render",
                    "--scene",
                    str(scene_root / "reconstruction/export/scene.json"),
                    "--cameras",
                    str(subrequest),
                    "--output",
                    str(out),
                ],
                check=True,
            )
            report[f"public_cli_{count}_seconds"] = time.perf_counter() - start
            shutil.rmtree(out)
            save()
        process = subprocess.Popen(
            [
                str(args.nht_python),
                str(args.nht_worker),
                "--scene",
                str(scene_root / "reconstruction/export/scene.json"),
                "--cameras",
                str(request),
                "--buffer",
                str(buffer),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

        assert process.stdin is not None

        def send(indices: list[int]) -> None:
            assert process.stdin is not None
            process.stdin.write(json.dumps({"op": "render", "indices": indices}) + "\n")
            process.stdin.flush()

        try:
            report["resident_startup"] = response(process)
            send(list(range(8)))
            report["resident_first"] = response(process)
            rgb = np.load(buffer)
            reference = np.stack([read_float32(r.image_path) for r in records[:8]])
            report["rerender_float_mae"] = float(np.abs(rgb - reference).mean())
            report["rerender_u8_different_fraction"] = float(
                np.mean(np.round(rgb * 255) != np.round(reference * 255))
            )
            timing = []
            for step in range(8):
                send(list(range(step * 8, step * 8 + 8)))
                timing.append(response(process))
            report["resident_batches"] = timing
            save()
            compressed = []
            for index, record in enumerate(records):
                path = Path(temporary) / f"{index}.f32.npz"
                write_float32(path, read_float32(record.image_path))
                compressed.append(dataclasses.replace(record, image_path=path))
            model = CourtDetectionLightningModule(
                cfg, target_bundle=dm.target_bundle_spec
            ).cuda()
            model.train()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1.0,
            )

            def make_cpu_batch(chosen: Any) -> Any:
                return court_detection_collate(
                    [pipeline.process(r) for r in chosen], bundle=dm.target_bundle_spec
                )

            def make_batch(chosen: Any) -> Any:
                return transfer(make_cpu_batch(chosen))

            def produce(indices: list[int]) -> Any:
                send(indices)
                response(process)
                frames = np.load(buffer)
                chosen = []
                for j, index in enumerate(indices):
                    path = Path(temporary) / f"live-{j}.npy"
                    np.save(path, frames[j], allow_pickle=False)
                    chosen.append(dataclasses.replace(records[index], image_path=path))
                return make_cpu_batch(chosen)

            def train(batch: Any) -> float:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    call = model.model_io.prepare_training_batch(batch)
                    output = model.model(*call.model_call.model_args)
                    result = model.model_io.training_result(output, call)
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                return float(result.loss.detach())

            cached = make_batch(records[:8])
            report["image_shape"] = list(cached["image"].shape)
            report["geometry"] = args.geometry
            for _ in range(4):
                train(cached)
            from torch.utils.data import DataLoader

            from src.tasks.court_detection.data.dataset import CourtDetectionDataset

            loader = DataLoader(
                CourtDetectionDataset(
                    records * (args.steps // 8 + 2), pipeline=pipeline
                ),
                batch_size=8,
                num_workers=4,
                pin_memory=True,
                collate_fn=partial(
                    court_detection_collate, bundle=dm.target_bundle_spec
                ),
            )
            loader_iterator = iter(loader)
            train(transfer(next(loader_iterator)))
            compressed_loader = DataLoader(
                CourtDetectionDataset(
                    tuple(compressed) * (args.steps // 8 + 2), pipeline=pipeline
                ),
                batch_size=8,
                num_workers=4,
                pin_memory=True,
                collate_fn=partial(
                    court_detection_collate, bundle=dm.target_bundle_spec
                ),
            )
            compressed_iterator = iter(compressed_loader)
            train(transfer(next(compressed_iterator)))
            executor = ThreadPoolExecutor(max_workers=1)
            measurements = {}
            # Same cameras, labels, loss and batch size. No convergence claim.
            for mode in [
                "cached_gpu",
                "npy_loader4",
                "compressed_loader4",
                "npy_pipeline",
                "compressed_pipeline",
                "ondemand_serial",
                "ondemand_prefetch",
                "ondemand_cpu_prefetch",
                "ondemand_reuse4",
            ]:
                torch.cuda.reset_peak_memory_stats()
                durations = []
                waits = []
                losses = []
                if mode == "ondemand_prefetch":
                    send(list(range(8)))
                if mode == "ondemand_cpu_prefetch":
                    future = executor.submit(produce, list(range(8)))
                reused = None
                for step in range(args.steps):
                    indices = list(range((step % 8) * 8, (step % 8) * 8 + 8))
                    start = time.perf_counter()
                    waited = 0.0
                    if mode == "ondemand_cpu_prefetch":
                        wait_start = time.perf_counter()
                        batch = transfer(future.result())
                        waited = time.perf_counter() - wait_start
                        if step + 1 < args.steps:
                            offset = ((step + 1) % 8) * 8
                            future = executor.submit(
                                produce, list(range(offset, offset + 8))
                            )
                    elif mode == "ondemand_reuse4":
                        if step % 4 == 0:
                            reused = transfer(produce(indices))
                        batch = reused
                    elif mode.startswith("ondemand"):
                        if mode == "ondemand_serial":
                            send(indices)
                        wait_start = time.perf_counter()
                        response(process)
                        waited = time.perf_counter() - wait_start
                        frames = np.load(buffer)
                        chosen = []
                        for j, index in enumerate(indices):
                            path = Path(temporary) / f"live-{j}.npy"
                            np.save(path, frames[j], allow_pickle=False)
                            chosen.append(
                                dataclasses.replace(records[index], image_path=path)
                            )
                        batch = make_batch(chosen)
                        if mode == "ondemand_prefetch" and step + 1 < args.steps:
                            offset = ((step + 1) % 8) * 8
                            send(list(range(offset, offset + 8)))
                    elif mode == "compressed_loader4":
                        batch = transfer(next(compressed_iterator))
                    elif mode == "npy_loader4":
                        batch = transfer(next(loader_iterator))
                    elif mode == "cached_gpu":
                        batch = cached
                    else:
                        collection = (
                            compressed if mode == "compressed_pipeline" else records
                        )
                        batch = make_batch([collection[i] for i in indices])
                    losses.append(train(batch))
                    durations.append(time.perf_counter() - start)
                    waits.append(waited)
                measurements[mode] = {
                    "step_seconds": durations,
                    "median_seconds": float(np.median(durations)),
                    "images_per_second": 8 * len(durations) / sum(durations),
                    "render_wait_seconds": waits,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                }
                report["training"] = measurements
                save()
                print(mode, measurements[mode]["images_per_second"], flush=True)
            executor.shutdown(wait=True)
        finally:
            if process.poll() is None:
                process.stdin.write('{"op":"stop"}\n')
                process.stdin.flush()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    save()


if __name__ == "__main__":
    main()
