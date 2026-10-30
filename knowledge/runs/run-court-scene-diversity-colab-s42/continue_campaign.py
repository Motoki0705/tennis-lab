"""Verify completed scene 2, then finish the pinned Colab campaign serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast

import colab_campaign as campaign
import yaml

REPO = Path(campaign.REPO)
EXPERIMENT = str(campaign.EXPERIMENT)
CORE_OBJECTS = {
    "src": "fc9a4761d9cfff3406453c89ce2987c0b6651333",
    "pyproject.toml": "852e0907312970530ff7c71b69215ef1a676cf9d",
    "uv.lock": "8d77c3e84d268d808872778ed1cdcf52738049c2",
    "knowledge/runs/run-court-scene-diversity-preparation-s42": "64888459df6c28dc3a175388769f5628c140e267",
    "knowledge/runs/run-court-scene-diversity-colab-s42/colab_campaign.py": "8225d640f3b8e382c7d848c3ad1e3f6c97c8a45f",
}
SCENE1_REPORT_SHA = "2ed28c9cf98e95ee82b8344632e4271ace255f509526dd61afc73c6764505a8a"


def run_root(condition: int) -> Path:
    return (
        REPO / "outputs/court_detection/train" / EXPERIMENT / f"scene-{condition}-s42"
    )


def evaluation_root(condition: int) -> Path:
    return (
        REPO
        / "outputs/court_detection/evaluate"
        / EXPERIMENT
        / f"scene-{condition}-s42"
    )


def expected_counts(condition: int) -> tuple[dict[str, int], dict[str, int]]:
    if condition not in range(1, 5):
        raise ValueError("Require condition 1–4.")
    val = {
        f"B{i:02d}": 500 // condition + int(i < 500 % condition)
        for i in range(condition)
    }
    return {scene: 6000 // condition - count for scene, count in val.items()}, val


def validate_config(
    expected: dict[str, Any], actual: dict[str, Any], condition: int
) -> set[str]:
    changes: set[str] = set(campaign.differences(expected, actual))
    if changes - {"run.resume"}:
        raise ValueError(
            f"Unexpected training configuration changes: {sorted(changes)}"
        )
    resume = actual["run"]["resume"]
    if resume is not None:
        if not isinstance(resume, dict) or set(resume) != {"role", "path"}:
            raise ValueError("Resume must be an explicit artifact reference.")
        path = PurePosixPath(resume["path"])
        prefix = PurePosixPath(
            f"court_detection/train/{EXPERIMENT}/scene-{condition}-s42"
        )
        if (
            resume["role"] != "artifact"
            or path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to(prefix)
            or path.suffix != ".ckpt"
        ):
            raise ValueError("Resume checkpoint belongs to another condition or root.")
    return changes


def validate_evaluation(
    report: dict[str, Any],
    condition: int,
    checkpoint_sha: str,
    data_condition: dict[str, Any],
    real_order: str,
) -> dict[str, Any]:
    from src.tasks.court_detection.ablation.evaluation import DENSE_METRICS

    if (
        report.get("schema") != "court_data_scaling_evaluation_v1"
        or report.get("series") != "scene-diversity"
        or report.get("global_step") != 33160
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("data_condition") != data_condition
    ):
        raise ValueError("Evaluation checkpoint or training identity changed.")
    train, val = expected_counts(condition)
    if (
        data_condition["scene_images"] != train
        or data_condition["synthetic_validation"]["scene_images"] != val
        or data_condition["accepted_images"] != 5500
        or data_condition["synthetic_validation"]["images"] != 500
        or data_condition["training_seed"] != 42
    ):
        raise ValueError("Evaluation per-scene counts or seed changed.")
    if set(report["splits"]) != {"synthetic_court-val", "tennis_court_detector-val"}:
        raise ValueError("Both separate validation domains are required.")
    result = {}
    for name, count, pose in [
        ("synthetic_court-val", 500, True),
        ("tennis_court_detector-val", 2211, False),
    ]:
        split = report["splits"][name]
        ids = split["sample_ids"]
        digest = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        expected = (
            data_condition["synthetic_validation"]["ordered_ids_sha256"]
            if pose
            else real_order
        )
        if (
            split["count"] != count
            or len(ids) != count
            or len(set(ids)) != count
            or digest != split["ordered_sample_ids_sha256"]
            or digest != expected
            or split["pose_supervised"] is not pose
            or split["batch_size"] != 8
        ):
            raise ValueError(
                "Validation count, order, batch size or supervision changed."
            )
        metrics = split["metrics"]
        if not set(DENSE_METRICS) <= metrics.keys() or (
            not pose and set(metrics) != set(DENSE_METRICS)
        ):
            raise ValueError(
                "Real validation must expose dense metrics without pose scoring."
            )
        if any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in metrics.values()
        ):
            raise ValueError("Evaluation metrics must be finite.")
        result[name] = {
            key: value
            for key, value in split.items()
            if key not in {"sample_ids", "qualitative"}
        }
    return result


def verify_core() -> None:
    for path, digest in CORE_OBJECTS.items():
        actual = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"], cwd=REPO, text=True
        ).strip()
        if actual != digest:
            raise ValueError(f"Pinned core source changed: {path}")


def final_checkpoint(config_dir: Path, audit: Path, condition: int) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    campaign.load_config(config_dir, condition)
    root = run_root(condition)
    completion = campaign.read(root / "training-completion.json")
    checkpoint = (REPO / completion["checkpoint"]).resolve(strict=True)
    if (
        not checkpoint.is_relative_to(root.resolve())
        or checkpoint.name != "update-00033160.ckpt"
        or campaign.sha(checkpoint) != completion["checkpoint_sha256"]
    ):
        raise ValueError("Final checkpoint path or content changed.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if type(payload["global_step"]) is not int or payload["global_step"] != 33160:
        raise ValueError("Expected actual final checkpoint global_step 33160.")
    hyper = payload["hyper_parameters"]
    actual = cast(
        dict[str, Any],
        OmegaConf.to_container(OmegaConf.create(hyper["config"]), resolve=True),
    )
    expected = yaml.safe_load((config_dir / f"scene-{condition}.yaml").read_text())
    changes = validate_config(expected, actual, condition)
    if actual != yaml.safe_load((root / "config.yaml").read_text()):
        raise ValueError("Checkpoint and saved run configurations disagree.")
    identity = campaign.read(root / "data-condition.json")
    if identity != hyper["data_condition"]:
        raise ValueError("Checkpoint and saved data conditions disagree.")
    value = {
        "schema": "court_scene_diversity_final_checkpoint_verification_v1",
        "condition": condition,
        "global_step": 33160,
        "checkpoint": completion["checkpoint"],
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "allowed_configuration_changes": sorted(changes),
        "resume": actual["run"]["resume"],
        "data_condition": identity,
    }
    campaign.write_fixed(audit / f"scene-{condition}-final-checkpoint.json", value)
    return value


def publish_verified(path: Path, receipt: Path) -> None:
    from src.utils.artifact_store import RcloneArtifactStore

    relative = path.relative_to(REPO).as_posix()
    store = RcloneArtifactStore(
        path,
        remote="gdrive",
        remote_root=f"tennis_lab/{relative}",
        config_path=Path(os.environ["RCLONE_CONFIG"]),
    )
    store.sync_tree()
    command = [
        sys.executable,
        str(REPO / ".agents/skills/tennis-drive/scripts/drive.py"),
        "verify",
        str(path),
        relative,
    ]
    with receipt.open("x") as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, check=True)
    if campaign.read(receipt).get("matches") is not True:
        raise ValueError("Drive verification did not confirm artifact bytes.")


def progress(audit: Path, phase: str, condition: int | None = None) -> None:
    value = {"phase": phase, "condition": condition}
    temporary = audit / "progress.tmp.json"
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(audit / "progress.json")
    print(json.dumps(value), flush=True)


def finish_evaluation(config_dir: Path, audit: Path, condition: int) -> dict[str, Any]:
    progress(audit, "final_checkpoint_verification", condition)
    checked = final_checkpoint(config_dir, audit, condition)
    publish_verified(
        run_root(condition), audit / f"scene-{condition}-training-drive.json"
    )
    target = evaluation_root(condition)
    if target.exists():
        raise FileExistsError(
            "Refusing to reuse an existing or partial evaluation directory."
        )
    progress(audit, "separate_domain_evaluation", condition)
    campaign.evaluate(condition)
    report = campaign.read(target / "evaluation.json")
    ready = campaign.read(config_dir / "data-readiness.json")
    splits = validate_evaluation(
        report,
        condition,
        checked["checkpoint_sha256"],
        checked["data_condition"],
        ready["real_validation_order_sha256"],
    )
    publish_verified(target, audit / f"scene-{condition}-evaluation-drive.json")
    result = {
        "schema": "court_scene_diversity_condition_result_v1",
        "condition": condition,
        "global_step": 33160,
        "checkpoint_sha256": checked["checkpoint_sha256"],
        "evaluation_sha256": campaign.sha(target / "evaluation.json"),
        "splits": splits,
    }
    campaign.write_fixed(audit / f"scene-{condition}-result.json", result)
    progress(audit, "evaluation_verified_and_saved", condition)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=campaign.DEFAULT_CONFIG_DIR)
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    audit.mkdir(parents=True, exist_ok=True)
    campaign.require_l4()
    verify_core()
    first = evaluation_root(1) / "evaluation.json"
    if campaign.sha(first) != SCENE1_REPORT_SHA:
        raise ValueError("Completed scene1 report changed; do not re-evaluate it.")
    finish_evaluation(args.config_dir, audit, 2)
    for condition in (3, 4):
        if any(run_root(condition).iterdir()):
            raise FileExistsError("Fresh conditions require an empty output directory.")
        config = campaign.load_config(args.config_dir, condition)
        if config.run.resume is not None or config.run.init_weights is not None:
            raise ValueError(
                "New conditions must initialize independently from the pinned pretrained backbone."
            )
        progress(audit, "fresh_independent_training", condition)
        campaign.train(args.config_dir, condition, None)
        finish_evaluation(args.config_dir, audit, condition)
    progress(audit, "comparison")
    target = REPO / "outputs/court_detection/compare" / EXPERIMENT / "s42"
    command = [
        sys.executable,
        str(campaign.PREP / "report.py"),
        "--evaluations",
        *[str(evaluation_root(n) / "evaluation.json") for n in range(1, 5)],
        "--output",
        str(target),
    ]
    subprocess.run(command, cwd=REPO, check=True)
    publish_verified(target, audit / "comparison-drive.json")
    campaign.write_fixed(
        audit / "completion.json",
        {
            "schema": "court_scene_diversity_continuation_v1",
            "status": "completed",
            "conditions": [1, 2, 3, 4],
            "core_objects": CORE_OBJECTS,
        },
    )
    progress(audit, "completed")


if __name__ == "__main__":
    main()
