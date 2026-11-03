"""Replay frozen clip inputs and compare repaired predictions against immutable v3."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.ball_trajectory.config import SegmentPhysicsConfig
from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import baseline_from_bundle
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison
from src.tennis_scene.evaluation.ball3d.scoring import score_comparison
from src.tennis_scene.evaluation.ball3d.segment_physics import SegmentPhysicsMethod
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def compare(left, right, path=""):
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        equal = (
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
        return [] if equal else [path]
    if is_dataclass(left) and is_dataclass(right):
        return [
            difference
            for field in fields(left)
            for difference in compare(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{path}.{field.name}",
            )
        ]
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [path + ".keys"]
        return [
            difference
            for key in left
            for difference in compare(left[key], right[key], f"{path}.{key}")
        ]
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [path + ".length"]
        return [
            difference
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for difference in compare(a, b, f"{path}[{index}]")
        ]
    return [] if left == right else [path]


def without_runtime(value):
    if isinstance(value, dict):
        return {
            key: without_runtime(item)
            for key, item in value.items()
            if key not in {"runtime_seconds_full_run", "runtime_seconds_all_four_runs"}
        }
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "annotations", "prior", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (args.prior, args.bundle, args.annotations.parent)
    ):
        raise ValueError("Use a new output separate from immutable inputs")
    script = Path(__file__)
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", str(script)], text=True
    ):
        raise ValueError("Commit solver and checker before recording the replay")
    prior_digest = dual_sha256(args.prior / "artifacts.json")
    if (
        prior_digest
        != "74dc43a3eb71598e3e95bd9ee4ec1effa8c1a97e739a0eb6b4732c922884d61e"
    ):
        raise ValueError("Expected the pinned immutable v3 inventory")
    inventory = json.loads((args.prior / "artifacts.json").read_text())
    for relative, declaration in inventory["files"].items():
        path = args.prior / relative
        if (
            path.stat().st_size != declaration["size_bytes"]
            or dual_sha256(path) != declaration["sha256"]
        ):
            raise ValueError(f"Prior artifact checksum mismatch: {relative}")
    bundle = load_bundle(
        args.bundle,
        expected_sha256="7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a",
    )
    if (
        bundle.inputs.clip_id != "meiji_3cam/video_000/clip_000"
        or bundle.inputs.num_frames != 1010
    ):
        raise ValueError("Restricted to the agreed clip")
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        config = compose(config_name="pipeline")
    values = OmegaConf.to_container(
        config.ball_reconstruction.segment_physics, resolve=True
    )
    if not isinstance(values, dict):
        raise ValueError("Composed physics configuration must be a mapping")
    settings = SegmentPhysicsConfig(**values)
    baseline = baseline_from_bundle(bundle)
    methods = {
        method.name: method
        for method in (
            baseline,
            SegmentPhysicsMethod(settings, baseline),
            SegmentPhysicsMethod(replace(settings, observation="heatmap"), baseline),
        )
    }
    prior_runs = {
        name: tuple(
            load_run(args.prior / "results" / name / fold)
            for fold in ("all_views", "without_cam0", "without_cam1", "without_cam2")
        )
        for name in methods
    }
    for name, method in methods.items():
        if dict(method.config) != prior_runs[name][0].config:
            raise ValueError("Composed method settings changed from v3")
    torch.set_num_threads(1)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest(
        {str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))}
    )
    print(
        "Pinned inputs/settings verified; replaying all methods and folds without labels",
        flush=True,
    )
    runs = run_comparison(
        methods,
        bundle.inputs,
        args.output / "results",
        bundle_sha256=bundle.manifest_sha256,
        code_commit=commit,
        code_fingerprint=fingerprint,
    )
    differences = {
        f"{name}/{run.key}": compare(prior.result, run.result)
        for name, values in runs.items()
        for prior, run in zip(prior_runs[name], values, strict=True)
    }
    print("Predictions saved; comparing and scoring", flush=True)
    annotations = load_annotations(
        args.annotations,
        bundle,
        expected_sha256="dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09",
    )
    runs["annotation_oracle"] = tuple(
        load_run(args.prior / "scoring_oracle" / fold)
        for fold in ("all_views", "without_cam0", "without_cam1", "without_cam2")
    )
    scores = score_comparison(runs, bundle, annotations)
    prior_scores = json.loads((args.prior / "scores.json").read_text())
    scoring_differences = compare(
        without_runtime(prior_scores), without_runtime(scores)
    )
    receipt = {
        "code_commit": commit,
        "prior_manifest_sha256": prior_digest,
        "verified_prior_files": len(inventory["files"]),
        "camera_order_in_each_fold": {
            run.key: list(run.result.camera_ids) for run in runs[baseline.name]
        },
        "config_equal_to_v3": True,
        "all_result_fields_bitwise_equal": not any(differences.values()),
        "result_differences": differences,
        "score_differences_excluding_runtime": scoring_differences,
        "runtime_fields_excluded_from_score_comparison": [
            "runtime_seconds_full_run",
            "runtime_seconds_all_four_runs",
        ],
        "normalized_prior_scores_sha256": document_digest(
            without_runtime(prior_scores)
        ),
        "normalized_current_scores_sha256": document_digest(without_runtime(scores)),
        "gpu_used": False,
        "detector_inference_rerun": False,
        "new_run_seconds": {
            name: sum(run.elapsed_seconds for run in values)
            for name, values in runs.items()
            if name != "annotation_oracle"
        },
    }
    write_json_atomic(args.output / "scores.json", scores)
    write_json_atomic(args.output / "comparison.json", receipt)
    print(json.dumps(receipt), flush=True)
    if any(differences.values()) or scoring_differences:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
