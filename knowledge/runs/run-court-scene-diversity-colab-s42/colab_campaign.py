"""Portable Colab execution of the pinned four-condition Court experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
PREP = REPO / "knowledge/runs/run-court-scene-diversity-preparation-s42"
DEFAULT_CONFIG_DIR = (
    REPO / "outputs/court_detection/campaign/scene-diversity-6000-colab-v1"
)
EXPERIMENT = "scene-diversity-6000-s384-colab"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_fixed(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    )
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to replace fixed evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def differences(a: Any, b: Any, prefix: str = "") -> set[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"Configuration keys changed at {prefix}")
        result: set[str] = set()
        for key in a:
            result |= differences(a[key], b[key], f"{prefix}.{key}".lstrip("."))
        return result
    return set() if a == b else {prefix}


def portable_config(
    original: dict[str, Any], condition: int, vm_root: Path
) -> dict[str, Any]:
    if condition not in (1, 2, 3, 4) or not vm_root.is_absolute():
        raise ValueError("Require an explicit condition and absolute VM root.")
    config = copy.deepcopy(original)
    config["paths"] = {
        "project_root": str(vm_root),
        "data_root": str(vm_root / "data"),
        "checkpoint_root": str(vm_root / "ckpt"),
        "artifact_root": str(vm_root / "outputs"),
        "output_root": str(vm_root / "outputs"),
        "external_asset_root": str(vm_root / "third_party"),
        "cache_root": str(vm_root / ".cache"),
    }
    output = f"court_detection/train/{EXPERIMENT}/scene-{condition}-s42"
    config["run"]["output_dir"] = output
    config["run"]["artifact_store"] = {
        "mode": "rclone",
        "remote": "gdrive",
        "remote_root": f"tennis_lab/outputs/{output}",
        "sync_interval_seconds": 60,
    }
    config["training"]["checkpoint"]["save_top_k"] = -1
    validate_conversion(original, config, condition, vm_root)
    return config


def validate_conversion(
    original: dict[str, Any], config: dict[str, Any], condition: int, vm_root: Path
) -> None:
    allowed = {f"paths.{key}" for key in original["paths"]} | {
        "run.output_dir",
        "run.artifact_store.mode",
        "run.artifact_store.remote",
        "run.artifact_store.remote_root",
        "run.artifact_store.sync_interval_seconds",
        "training.checkpoint.save_top_k",
    }
    changed = differences(original, config)
    if changed - allowed:
        raise ValueError(
            f"Training semantics changed during relocation: {sorted(changed - allowed)}"
        )
    expected = copy.deepcopy(original)
    expected["paths"] = {
        "project_root": str(vm_root),
        "data_root": str(vm_root / "data"),
        "checkpoint_root": str(vm_root / "ckpt"),
        "artifact_root": str(vm_root / "outputs"),
        "output_root": str(vm_root / "outputs"),
        "external_asset_root": str(vm_root / "third_party"),
        "cache_root": str(vm_root / ".cache"),
    }
    output = f"court_detection/train/{EXPERIMENT}/scene-{condition}-s42"
    expected["run"]["output_dir"] = output
    expected["run"]["artifact_store"] = {
        "mode": "rclone",
        "remote": "gdrive",
        "remote_root": f"tennis_lab/outputs/{output}",
        "sync_interval_seconds": 60,
    }
    expected["training"]["checkpoint"]["save_top_k"] = -1
    if (
        config != expected
        or config["run"]["resume"] is not None
        or config["run"]["init_weights"] is not None
    ):
        raise ValueError(
            "Portable config differs from the explicit VM/storage mapping."
        )


def prepare(config_dir: Path, vm_root: Path) -> None:
    plan = read(PREP / "plan.json")
    configs = {}
    config_dir.mkdir(parents=True, exist_ok=True)
    for condition in range(1, 5):
        source = PREP / f"scene-{condition}.yaml"
        if sha(source) != plan["config_sha256"][str(condition)]:
            raise ValueError("Original prepared config changed.")
        original = yaml.safe_load(source.read_text())
        converted = portable_config(original, condition, vm_root)
        target = config_dir / source.name
        content = yaml.safe_dump(converted, allow_unicode=True, sort_keys=False)
        if target.exists() and target.read_text() != content:
            raise ValueError("Refusing to overwrite a portable condition.")
        target.write_text(content)
        configs[str(condition)] = {
            "original_sha256": sha(source),
            "portable_sha256": sha(target),
            "changed_fields": sorted(differences(original, converted)),
        }
    write_fixed(
        config_dir / "portable-plan.json",
        {
            "schema": "court_scene_diversity_colab_plan_v1",
            "source_plan_sha256": sha(PREP / "plan.json"),
            "source_readiness_sha256": sha(PREP / "data-readiness.json"),
            "vm_root": str(vm_root),
            "conditions": configs,
            "checkpoint_retention_change": "save_top_k: 0 -> -1; all generations retained; final selection remains update 33160",
        },
    )
    print(json.dumps({"prepared": 4, "config_dir": str(config_dir)}), flush=True)


def load_config(config_dir: Path, condition: int):
    from omegaconf import OmegaConf

    from src.tasks.court_detection.scaling.schedule import StepSchedule
    from src.tasks.court_detection.training.runner import resolve_training_config

    pinned = read(config_dir / "portable-plan.json")
    path = config_dir / f"scene-{condition}.yaml"
    if sha(path) != pinned["conditions"][str(condition)]["portable_sha256"]:
        raise ValueError("Portable configuration changed after preparation.")
    original = yaml.safe_load((PREP / path.name).read_text())
    validate_conversion(
        original, yaml.safe_load(path.read_text()), condition, Path(pinned["vm_root"])
    )
    config = OmegaConf.load(path)
    resolve_training_config(config)
    StepSchedule.from_config(config)
    return config


def preflight(config_dir: Path) -> None:
    from src.tasks.court_detection.configuration import CourtTrainingConfig
    from src.tasks.court_detection.data.inputs.factory import build_court_input
    from src.tasks.court_detection.scaling.real_split import (
        TennisCourtEvaluationInput,
        canonical_sha256,
    )
    from src.tasks.court_detection.training.runner import resolve_training_config

    plan = read(PREP / "plan.json")
    ready = read(PREP / "data-readiness.json")
    prior = read(
        REPO / "knowledge/runs/run-i983-court-dinov3-s-s42/evaluation/evaluation.json"
    )
    expected_order = prior["splits"]["tennis_court_detector-val"][
        "ordered_sample_ids_sha256"
    ]
    conditions = []
    all_groups = {"train": set(), "val": set()}
    for n in range(1, 5):
        config = load_config(config_dir, n)
        runtime = CourtTrainingConfig.from_config(config)
        _, mixed = resolve_training_config(config)
        real_config = mixed.sources["tennis_court_detector"]
        real = TennisCourtEvaluationInput(real_config)
        if (
            len(real.records("val")) != 2211
            or sha(real_config.registry) != plan["real_registry_sha256"]
        ):
            raise ValueError("Original real validation registry changed.")
        registry = read(real_config.registry)
        if canonical_sha256(registry["splits"]["val"]) != expected_order:
            raise ValueError("Real validation order differs from #983.")
        if sha(runtime.model.encoder.checkpoint_path) != plan["pretrained_sha256"]:
            raise ValueError("Initial DINOv3-S weight changed.")
        source = build_court_input(runtime.data.source)
        condition = {
            "condition": n,
            "pool_sha256": source.version_manifest_sha256,
            "counts": {},
        }
        if source.version_manifest_sha256 != ready["conditions"][n - 1]["pool_sha256"]:
            raise ValueError(
                "Synthetic pool differs from the generated frozen version."
            )
        for split in ("train", "val"):
            rows = source.records(split)
            counts = Counter(str(r.payload["scene_id"]) for r in rows)
            expected_counts = plan[
                "train_scene_counts" if split == "train" else "val_scene_counts"
            ][n - 1]
            if counts != expected_counts:
                raise ValueError("Synthetic per-scene counts changed.")
            manifest = (
                runtime.data.source.manifest
                if split == "train"
                else runtime.data.source.validation_manifest
            )
            if sha(manifest) != ready["conditions"][n - 1]["subset_sha256"][split]:
                raise ValueError("Selected synthetic subset changed.")
            for record in rows:
                all_groups[split].add(
                    (
                        str(record.payload["scene_id"]),
                        record.payload["trajectory_group_id"],
                    )
                )
            condition["counts"][split] = dict(counts)
        conditions.append(condition)
        print(json.dumps({"preflight_condition": condition}), flush=True)
    if all_groups["train"] & all_groups["val"]:
        raise ValueError("Cross-condition train/validation trajectory leakage.")
    write_fixed(
        config_dir / "data-readiness.json",
        {
            "schema": "court_scene_diversity_colab_ready_v1",
            "portable_plan_sha256": sha(config_dir / "portable-plan.json"),
            "real_validation_order_sha256": expected_order,
            "real_registry_sha256": plan["real_registry_sha256"],
            "pretrained_sha256": plan["pretrained_sha256"],
            "conditions": conditions,
        },
    )


def require_l4() -> None:
    import torch

    if (
        Path("/content/tennis-lab") != REPO
        or not torch.cuda.is_available()
        or "L4" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError(
            "This execution entry requires the owned /content/tennis-lab Colab L4 checkout."
        )


def probe(config_dir: Path) -> None:
    import pytorch_lightning as pl
    import torch

    from src.tasks.base.training.artifact_store import build_artifact_store
    from src.tasks.court_detection.configuration import CourtTrainingConfig
    from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner

    require_l4()
    config = load_config(config_dir, 1)
    if not (config_dir / "data-readiness.json").is_file():
        raise ValueError("Verified Colab inputs are required before the GPU probe.")
    config.training.step_schedule.max_steps = 3
    config.training.step_schedule.validation_every_n_steps = 3
    config.training.step_schedule.checkpoint_every_n_steps = 3
    config.training.warmup_steps = 0
    config.run.output_dir = f"court_detection/train/{EXPERIMENT}-probe/scene-1-s42"
    config.run.artifact_store.remote_root = (
        f"tennis_lab/outputs/{config.run.output_dir}"
    )
    evidence: dict[str, Any] = {
        "schema": "court_scene_diversity_colab_probe_v1",
        "is_full_training_result": False,
        "gpu": torch.cuda.get_device_name(0),
        "portable_plan_sha256": sha(config_dir / "portable-plan.json"),
        "ready_sha256": sha(config_dir / "data-readiness.json"),
        "probe_overrides": {
            "max_steps": 3,
            "validation_every_n_steps": 3,
            "checkpoint_every_n_steps": 3,
            "warmup_steps": 0,
        },
        "steps": [],
    }

    class Recorder(pl.Callback):
        def on_train_start(self, trainer, module):
            torch.cuda.reset_peak_memory_stats()

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if batch["image"].shape[0] != 8 or not batch["pose_supervision_mask"].all():
                raise ValueError(
                    "GPU probe requires the eight synthetic production-shape images."
                )
            torch.cuda.synchronize()
            self.start = time.monotonic()
            self.shape = list(batch["image"].shape)

        def on_before_optimizer_step(self, trainer, module, optimizer):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("Non-finite probe gradient.")

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("Non-finite probe loss.")
            torch.cuda.synchronize()
            item = {
                "step": trainer.global_step,
                "loss": value,
                "seconds": time.monotonic() - self.start,
                "image_shape": self.shape,
            }
            evidence["steps"].append(item)
            print(json.dumps({"probe_step": item}), flush=True)

    class ProbeRunner(CourtDetectionTrainingRunner):
        def callbacks_extra(self, config, datamodule, logger):
            return [*super().callbacks_extra(config, datamodule, logger), Recorder()]

    ProbeRunner().run(config)
    runtime = CourtTrainingConfig.from_config(config)
    checkpoints = list(
        runtime.shared.run.output_dir.glob(
            "logs/version_*/checkpoints/update-00000003.ckpt"
        )
    )
    if len(checkpoints) != 1 or len(evidence["steps"]) != 3:
        raise ValueError(
            "Probe did not complete and save exactly three optimizer updates."
        )
    evidence.update(
        status="completed",
        checkpoint=str(checkpoints[0]),
        checkpoint_sha256=sha(checkpoints[0]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    write_fixed(config_dir / "gpu-probe.json", evidence)
    write_fixed(runtime.shared.run.output_dir / "gpu-probe.json", evidence)
    store = build_artifact_store(
        runtime.shared.run.artifact_store, local_root=runtime.shared.run.output_dir
    )
    store.publish_file(runtime.shared.run.output_dir / "gpu-probe.json")


def train(config_dir: Path, condition: int, resume: Path | None) -> None:
    require_l4()
    load_config(config_dir, condition)
    probe_result = read(config_dir / "gpu-probe.json")
    if (
        probe_result["status"] != "completed"
        or probe_result["portable_plan_sha256"]
        != sha(config_dir / "portable-plan.json")
        or probe_result["ready_sha256"] != sha(config_dir / "data-readiness.json")
    ):
        raise ValueError("A successful probe of these pinned Colab inputs is required.")
    output = (
        REPO / "outputs" / f"court_detection/train/{EXPERIMENT}/scene-{condition}-s42"
    )
    if list(output.glob("logs/version_*/checkpoints/update-00033160.ckpt")):
        raise ValueError(
            "This condition is already trained to the final update; evaluate it."
        )
    command = [
        sys.executable,
        "-m",
        "src.tasks.court_detection.scripts.train",
        "--config-path",
        str(config_dir),
        "--config-name",
        f"scene-{condition}",
    ]
    if resume is not None:
        checkpoint = resume.resolve(strict=True)
        relative = checkpoint.relative_to(REPO / "outputs").as_posix()
        if not checkpoint.is_relative_to(output):
            raise ValueError(
                "Resume checkpoint must belong to this same condition/run."
            )
        command.append(f"run.resume={{role:artifact,path:{relative}}}")
    subprocess.run(
        command,
        cwd=REPO,
        check=True,
        env={
            **os.environ,
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    final = list(output.glob("logs/version_*/checkpoints/update-00033160.ckpt"))
    if len(final) != 1:
        raise ValueError("Expected exactly one checkpoint at update 33160.")
    write_fixed(
        output / "training-completion.json",
        {
            "schema": "court_scene_diversity_colab_training_completion_v1",
            "condition": condition,
            "global_step": 33160,
            "checkpoint": final[0].relative_to(REPO).as_posix(),
            "checkpoint_sha256": sha(final[0]),
            "resume_checkpoint": None if resume is None else str(resume),
        },
    )


def evaluate(condition: int) -> None:
    require_l4()
    output = (
        REPO / "outputs" / f"court_detection/train/{EXPERIMENT}/scene-{condition}-s42"
    )
    completion = read(output / "training-completion.json")
    checkpoint = REPO / completion["checkpoint"]
    if sha(checkpoint) != completion["checkpoint_sha256"]:
        raise ValueError("Completed final checkpoint changed.")
    target = (
        REPO
        / "outputs"
        / f"court_detection/evaluate/{EXPERIMENT}/scene-{condition}-s42"
    )
    command = [
        sys.executable,
        "-m",
        "src.tasks.court_detection.scripts.evaluate_ablation",
        "--checkpoint",
        str(checkpoint),
        "--expected-step",
        "33160",
        "--split",
        "val",
        "--series",
        "scene-diversity",
        "--include-synthetic-val",
        "--output",
        str(target),
        "--data-root",
        str(REPO / "data"),
        "--checkpoint-root",
        str(REPO / "ckpt"),
        "--external-asset-root",
        str(REPO / "third_party"),
        "--artifact-root",
        str(REPO / "outputs"),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "preflight", "probe", "train", "evaluate")
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--vm-root", type=Path, default=Path("/content/tennis-lab"))
    parser.add_argument("--condition", type=int, choices=range(1, 5))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config_dir, args.vm_root)
    elif args.action == "preflight":
        preflight(args.config_dir)
    elif args.action == "probe":
        probe(args.config_dir)
    elif args.action == "train" and args.condition is not None:
        train(args.config_dir, args.condition, args.resume)
    elif args.action == "evaluate" and args.condition is not None:
        evaluate(args.condition)
    else:
        parser.error("--condition is required for train/evaluate")


if __name__ == "__main__":
    main()
