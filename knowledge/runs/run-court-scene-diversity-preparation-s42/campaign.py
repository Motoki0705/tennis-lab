"""Prepare, verify and enqueue the fixed-budget Court scene-diversity experiment."""

from __future__ import annotations

# Standalone reproduction scripts bootstrap this checkout before importing src.
# ruff: noqa: E402
import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.scaling.real_split import (
    TennisCourtEvaluationInput,
    canonical_sha256,
)
from src.tasks.court_detection.scaling.schedule import StepSchedule
from src.tasks.court_detection.training.runner import resolve_training_config
from src.utils.data.image_record_store import file_sha256

MAIN = (
    Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=REPO, text=True
        ).strip()
    )
    .resolve()
    .parent
)
CAMPAIGN = MAIN / "outputs/court_detection/campaign/scene-diversity-6000-v1"
DATA = MAIN / "data/court_scene_diversity_6000_v1"
BUDGET = 33160
TRAIN_COUNTS = (
    {"B00": 5500},
    {"B00": 2750, "B01": 2750},
    {"B00": 1833, "B01": 1833, "B02": 1834},
    {"B00": 1375, "B01": 1375, "B02": 1375, "B03": 1375},
)
VAL_COUNTS = (
    {"B00": 500},
    {"B00": 250, "B01": 250},
    {"B00": 167, "B01": 167, "B02": 166},
    {"B00": 125, "B01": 125, "B02": 125, "B03": 125},
)


def config_path(condition):
    return CAMPAIGN / "configs" / f"scene-{condition}.yaml"


def load_config(condition):
    config = OmegaConf.load(config_path(condition))
    resolve_training_config(config)
    schedule = StepSchedule.from_config(config)
    runtime = CourtTrainingConfig.from_config(config)
    model = runtime.model
    if (
        schedule.max_steps != BUDGET
        or config.data.batch_size != 8
        or config.run.seed != 42
        or config.run.init_weights is not None
        or config.run.resume is not None
        or model.encoder.backbone_name != "dinov3_vits16"
        or model.encoder.train_mode != "frozen"
        or model.encoder.lora.enabled
        or (
            model.transformer_encoder.dim,
            model.transformer_encoder.depth,
            model.transformer_encoder.num_heads,
            model.transformer_encoder.ffn_dim,
        )
        != (384, 8, 6, 1024)
        or config.mixed.train_batch_counts.synthetic_court != 8
        or config.mixed.train_batch_counts.tennis_court_detector != 0
    ):
        raise ValueError("Prepared configuration differs from the agreed experiment.")
    return config


def prepare():
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    registry = CAMPAIGN / "evaluation/registry.json"
    real = json.loads(registry.read_text())
    prior = json.loads(
        (
            REPO
            / "knowledge/runs/run-i983-court-dinov3-s-s42/evaluation/evaluation.json"
        ).read_text()
    )
    prior_identity = prior["splits"]["tennis_court_detector-val"][
        "ordered_sample_ids_sha256"
    ]
    if (
        len(real["splits"]["val"]) != 2211
        or canonical_sha256(real["splits"]["val"]) != prior_identity
    ):
        raise ValueError("Original validation differs from the previous size ablation.")
    overrides = [
        "data/source=synthetic_court_dataset_subset",
        f"paths.project_root={REPO}",
        f"paths.data_root={MAIN / 'data'}",
        f"paths.checkpoint_root={MAIN / 'ckpt'}",
        f"paths.artifact_root={MAIN / 'outputs'}",
        f"paths.output_root={MAIN / 'outputs'}",
        f"paths.external_asset_root={MAIN / 'third_party'}",
        f"paths.cache_root={MAIN / '.cache'}",
        "model.encoder.backbone_name=dinov3_vits16",
        "model.encoder.checkpoint_path=dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "model.transformer_encoder.dim=384",
        "model.transformer_encoder.num_heads=6",
        "model.transformer_encoder.ffn_dim=1024",
        "mixed.sources.tennis_court_detector.registry=court_detection/campaign/scene-diversity-6000-v1/evaluation/registry.json",
        "training.warmup_steps=4974",
        "training.step_schedule.max_steps=33160",
        "training.step_schedule.validation_every_n_steps=3316",
        "training.step_schedule.checkpoint_every_n_steps=3316",
        "training.checkpoint.save_top_k=0",
    ]
    (CAMPAIGN / "configs").mkdir(exist_ok=True)
    config_hashes = {}
    for condition in range(1, 5):
        with initialize_config_dir(
            config_dir=str(REPO / "src/tasks/court_detection/configs"),
            version_base="1.3",
        ):
            config = compose(
                config_name="train_data_scaling",
                overrides=[
                    *overrides,
                    f"data.source.manifest=court_scene_diversity_6000_v1/scene-{condition}.train.subset.json",
                    f"data.source.validation_manifest=court_scene_diversity_6000_v1/scene-{condition}.val.subset.json",
                    f"run.output_dir=court_detection/train/scene-diversity-6000-s384/scene-{condition}-s42",
                ],
            )
        resolve_training_config(config)
        StepSchedule.from_config(config)
        target = config_path(condition)
        content = OmegaConf.to_yaml(config, resolve=True)
        if target.exists() and target.read_text() != content:
            raise ValueError(f"Refusing to replace prepared config: {target}")
        target.write_text(content)
        config_hashes[str(condition)] = file_sha256(target)
    model_path = MAIN / "ckpt/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    plan = {
        "schema": "court_scene_diversity_campaign_v1",
        "steps": BUDGET,
        "train_images": 5500,
        "synthetic_val_images": 500,
        "real_val_images": 2211,
        "training_seed": 42,
        "train_scene_counts": list(TRAIN_COUNTS),
        "val_scene_counts": list(VAL_COUNTS),
        "config_sha256": config_hashes,
        "real_registry_sha256": file_sha256(registry),
        "pretrained_sha256": file_sha256(model_path),
    }
    plan_path = CAMPAIGN / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Refusing to change the pinned campaign plan.")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prepared": 4,
                "campaign": str(CAMPAIGN),
                "data_ready": all(
                    (DATA / f"scene-{n}.{s}.subset.json").exists()
                    for n in range(1, 5)
                    for s in ("train", "val")
                ),
            }
        )
    )


def preflight():
    plan = json.loads((CAMPAIGN / "plan.json").read_text())
    actual = []
    pools = set()
    groups = {}
    previous = {}
    for condition in range(1, 5):
        path = config_path(condition)
        if file_sha256(path) != plan["config_sha256"][str(condition)]:
            raise ValueError("Prepared config changed.")
        config = load_config(condition)
        runtime = CourtTrainingConfig.from_config(config)
        _, mixed = resolve_training_config(config)
        real = TennisCourtEvaluationInput(mixed.sources["tennis_court_detector"])
        if (
            len(real.records("val")) != 2211
            or file_sha256(mixed.sources["tennis_court_detector"].registry)
            != plan["real_registry_sha256"]
        ):
            raise ValueError("Real validation inventory changed.")
        if (
            file_sha256(runtime.model.encoder.checkpoint_path)
            != plan["pretrained_sha256"]
        ):
            raise ValueError("Pretrained DINO weight changed.")
        source = build_court_input(runtime.data.source)
        pools.add(source.version_manifest_sha256)
        for split, expected in [
            ("train", TRAIN_COUNTS[condition - 1]),
            ("val", VAL_COUNTS[condition - 1]),
        ]:
            records = source.records(split)
            counts = Counter(str(r.payload["scene_id"]) for r in records)
            if counts != expected:
                raise ValueError(
                    f"Condition {condition} {split}: {counts} != {expected}"
                )
            for scene in counts:
                ids = {r.sample_id for r in records if r.payload["scene_id"] == scene}
                key = (split, scene)
                if key in previous and not ids <= previous[key]:
                    raise ValueError(
                        "Scene selections are not nested across conditions."
                    )
                previous[key] = ids
            for record in records:
                group = (
                    record.payload["scene_id"],
                    record.payload["trajectory_group_id"],
                )
                if group in groups and groups[group] != split:
                    raise ValueError("Trajectory changes split across conditions.")
                groups[group] = split
        actual.append(
            {
                "condition": condition,
                "subset_sha256": source.subset_manifest_sha256,
                "pool_sha256": source.version_manifest_sha256,
            }
        )
    if len(pools) != 1:
        raise ValueError("Conditions must reference one immutable pool.")
    report = {
        "schema": "court_scene_diversity_ready_v1",
        "conditions": actual,
        "plan_sha256": file_sha256(CAMPAIGN / "plan.json"),
    }
    ready_path = CAMPAIGN / "ready.json"
    if ready_path.exists() and json.loads(ready_path.read_text()) != report:
        raise ValueError("The validated dataset changed after readiness was recorded.")
    ready_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ready": True, "conditions": 4}))
    return report


def require_gpu_probe():
    probe = json.loads((CAMPAIGN / "gpu-probe.json").read_text())
    if (
        probe["status"] != "completed"
        or probe["plan_sha256"] != file_sha256(CAMPAIGN / "plan.json")
        or probe["ready_sha256"] != file_sha256(CAMPAIGN / "ready.json")
    ):
        raise ValueError("A successful GPU probe of the pinned inputs is required.")


def execute(condition):
    # This entry is launched only by the shared GPU queue.
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError(
            "Execute this GPU workload through the shared training queue."
        )
    preflight()
    require_gpu_probe()
    config = load_config(condition)
    command = [
        sys.executable,
        "-m",
        "src.tasks.court_detection.scripts.train",
        "--config-path",
        str(CAMPAIGN / "configs"),
        "--config-name",
        f"scene-{condition}",
    ]
    subprocess.run(command, cwd=REPO, check=True)
    runtime = CourtTrainingConfig.from_config(config)
    candidates = list(
        runtime.shared.run.output_dir.glob(
            "logs/version_*/checkpoints/update-00033160.ckpt"
        )
    )
    if len(candidates) != 1:
        raise ValueError("Expected exactly one checkpoint at update 33160.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.court_detection.scripts.evaluate_ablation",
            "--checkpoint",
            str(candidates[0]),
            "--expected-step",
            str(BUDGET),
            "--split",
            "val",
            "--series",
            "scene-diversity",
            "--include-synthetic-val",
            "--output",
            str(CAMPAIGN / "evaluations" / f"scene-{condition}"),
        ],
        cwd=REPO,
        check=True,
    )


def enqueue(session):
    preflight()
    require_gpu_probe()
    environment = dict(os.environ, TRAINING_QUEUE_DIR=str(MAIN / ".training_queue"))
    queue = REPO / ".agents/skills/training-queue/scripts/training_queue.sh"
    for condition in range(1, 5):
        command = shlex.join(
            [
                "env",
                "TORCHINDUCTOR_COMPILE_THREADS=1",
                str(REPO / ".venv/bin/python"),
                str(Path(__file__).resolve()),
                "execute",
                "--condition",
                str(condition),
            ]
        )
        subprocess.run(
            [
                "bash",
                str(queue),
                "add",
                command,
                "--resource",
                "all",
                "--name",
                f"court_scene_diversity_{condition}_s42",
                "--provider",
                "codex",
                "--session",
                session,
                "--idempotency-key",
                f"court-scene-diversity-v1/scene-{condition}/attempt-1",
            ],
            cwd=REPO,
            env=environment,
            check=True,
        )
    subprocess.run(["bash", str(queue), "start"], cwd=REPO, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "preflight", "enqueue", "execute"]
    )
    parser.add_argument("--condition", type=int, choices=range(1, 5))
    parser.add_argument("--session")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "preflight":
        preflight()
    elif args.action == "enqueue":
        if not args.session:
            parser.error("--session is required to attribute queue jobs")
        enqueue(args.session)
    else:
        if args.condition is None:
            parser.error("--condition is required")
        execute(args.condition)


if __name__ == "__main__":
    main()
