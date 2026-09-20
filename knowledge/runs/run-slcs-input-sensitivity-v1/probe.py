"""Fixed archived train batches; CPU eval input-time sensitivity, no training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.training.losses import build_slcs_loss_inputs
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

PREVIOUS = Path("knowledge/runs/run-slcs-ball-gradient-probe-v1")
spec = importlib.util.spec_from_file_location("gradient_probe", PREVIOUS / "probe.py")
assert spec is not None and spec.loader is not None
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def intervene(
    batch: dict[str, torch.Tensor], condition: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Reverse active slots, keeping padding and all targets/frame slots fixed.

    Ball visibility is the input confidence/validity signal: transport it with
    its UV observation. DINO padding stays fixed; reverse only real samples.
    """
    result = dict(batch)
    mapping: dict[str, Any] = {}
    keys: tuple[str, ...]
    if condition == "ball_reverse":
        active = (~batch["padding_mask"][0]).nonzero().flatten()
        keys = ("ball_uv", "ball_vis")
    elif condition == "dino_reverse":
        active = (~batch["dino_padding_mask"][0]).nonzero().flatten()
        keys = ("dino_tokens",)
    elif condition == "full":
        return result, mapping
    else:
        raise ValueError(condition)
    if batch["padding_mask"].shape[0] != 1:
        raise ValueError("This probe requires batch size 1")
    source = active.flip(0)
    for key in keys:
        result[key] = batch[key].clone()
        result[key][:, active] = batch[key][:, source]
    mapping = {"destination_slots": active.tolist(), "source_slots": source.tolist()}
    return result, mapping


def metrics(
    pred: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    real: torch.Tensor,
    teacher: torch.Tensor,
) -> dict[str, Any]:
    """Float64 metric arithmetic, normalized court -> physical meters once."""
    scale = torch.tensor(COURT_COORD_SCALE_XYZ, dtype=torch.float64)
    delta = (pred.double() - baseline.double()) * scale
    distance = delta[real].norm(dim=-1)
    error = ((pred.double() - target.double()) * scale)[teacher].norm(dim=-1)
    return {
        "output_difference_m": {
            "rms_euclidean": float(distance.square().mean().sqrt()),
            "mean_euclidean": float(distance.mean()),
            "max_euclidean": float(distance.max()),
            "rms_xyz": delta[real].square().mean(dim=0).sqrt().tolist(),
        },
        "teacher_error_m_mean": float(error.mean()),
        "real_frame_count": int(real.sum()),
        "teacher_frame_count": int(teacher.sum()),
    }


def run(output: Path, report: dict[str, Any]) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    old = json.loads((PREVIOUS / "results.json").read_text())
    report.update(
        {
            "seed": 42,
            "mode": "eval",
            "device": "cpu",
            "precision": "float32",
            "threads": 2,
            "augmentation": False,
            "optimizer_steps": 0,
            "scope": "Two fixed archived train windows; OOD input-time interventions, not robustness or quality proof; no val/test inference.",
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "court_coordinate_scale_xyz": list(COURT_COORD_SCALE_XYZ),
            "source_sha256": {},
            "domains": {},
            "intervention": {
                "ball_reverse": "Reverse UV and associated ball_vis jointly over nonpadding frame slots; targets and timestamps unchanged.",
                "dino_reverse": "Reverse nonpadding DINO token samples; absolute dino_frame_idx, dino_padding_mask, frame_idx, and targets unchanged.",
            },
        }
    )
    sources = [
        prior.CONFIG,
        prior.CHECKPOINT,
        PREVIOUS / "results.json",
        PREVIOUS / "probe.py",
        Path(__file__),
    ]
    for path in sources:
        report["stage"] = f"hash:{path}"
        prior.publish_results(output, report)
        report["source_sha256"][str(path)] = dual_sha256(path)
    if report["source_sha256"][str(prior.CHECKPOINT)] != prior.EXPECTED_SHA:
        raise ValueError("Checkpoint hash differs from validation-selected epoch55")
    if (
        report["source_sha256"][str(prior.CONFIG)]
        != old["input_sha256"][str(prior.CONFIG)]
    ):
        raise ValueError("Training config changed from prior probe")
    report["checkpoint_sha256"] = prior.EXPECTED_SHA
    config = OmegaConf.load(prior.CONFIG)
    if not isinstance(config, DictConfig):
        raise TypeError("Training config must be a mapping")
    runtime = SLCSTrainingRuntimeConfig.from_config(config)
    resolver = PathResolver(
        replace(
            runtime.resolver.roots, checkpoint_root=prior.CHECKPOINT.parent.resolve()
        )
    )
    report["stage"] = "load_checkpoint"
    prior.publish_results(output, report)
    predictor = SLCSPredictor.load_from_checkpoint(
        prior.CHECKPOINT,
        resolver=resolver,
        device="cpu",
        strict=True,
        weights_only=False,
    )
    for section in ("model", "data", "loss"):
        if OmegaConf.to_container(
            predictor.lightning_module.config[section], resolve=True
        ) != OmegaConf.to_container(config[section], resolve=True):
            raise ValueError(f"Checkpoint config mismatch: {section}")
    OmegaConf.save(config, output / "resolved_training_config.yaml", resolve=True)
    predictor.model.eval()
    before_state = {
        k: prior.tensor_sha(v) for k, v in predictor.model.state_dict().items()
    }
    for domain in ("broadcast", "meiji"):
        source = (
            prior.MAIN
            / "outputs/slcs/analyze/ball_gradient_probe/s42-001"
            / f"{domain}_inputs.npz"
        )
        report["stage"] = f"hash:{source}"
        prior.publish_results(output, report)
        report["source_sha256"][str(source)] = dual_sha256(source)
        with np.load(source, allow_pickle=False) as archive:
            batch = {
                k: torch.from_numpy(archive[k])
                for k in old["domains"][domain]["input_tensor_sha256"]
            }
        identities = {k: prior.tensor_sha(v) for k, v in batch.items()}
        if identities != old["domains"][domain]["input_tensor_sha256"]:
            raise ValueError(f"Archived input identity mismatch: {domain}")
        record = {k: v for k, v in old["domains"][domain].items() if k != "modes"}
        record["conditions"] = {}
        report["domains"][domain] = record
        arrays = {k: v.numpy() for k, v in batch.items() if k != "dino_tokens"}
        baseline = None
        for condition in ("full", "ball_reverse", "dino_reverse"):
            report["stage"] = f"{domain}/{condition}"
            prior.publish_results(output, report)
            changed, mapping = intervene(batch, condition)
            torch.manual_seed(42)
            with torch.inference_mode():
                targets = predictor.model_adapter.build_training_targets(changed)
                inputs = build_slcs_loss_inputs(
                    predictor.model_io.run(changed), targets
                )
                pred = inputs.pred_ball_position
                prior.finite(pred, condition)
                if baseline is None:
                    baseline = pred.clone()
                result = metrics(
                    pred,
                    baseline,
                    inputs.target_ball_position,
                    ~inputs.padding_mask,
                    inputs.ball_mask,
                )
                result["mapping"] = mapping
                result["changed_tensor_keys"] = [
                    k
                    for k, v in changed.items()
                    if prior.tensor_sha(v) != identities[k]
                ]
                result["input_tensor_sha256"] = {
                    k: prior.tensor_sha(v) for k, v in changed.items()
                }
                record["conditions"][condition] = result
                arrays[f"{condition}_pred_ball_position_normalized"] = pred.numpy()
                arrays["metric_teacher_mask"] = inputs.ball_mask.numpy()
                arrays[f"{condition}_ball_uv"] = changed["ball_uv"].numpy()
                arrays[f"{condition}_ball_vis"] = changed["ball_vis"].numpy()
            if {k: prior.tensor_sha(v) for k, v in batch.items()} != identities:
                raise ValueError("Original batch mutated")
        assert baseline is not None
        with np.load(PREVIOUS / f"{domain}_eval.npz", allow_pickle=False) as reference:
            delta = np.max(np.abs(baseline.numpy() - reference["pred_ball_position"]))
        record["prior_eval_max_abs_normalized_difference"] = float(delta)
        if delta > 1e-6:
            raise ValueError(f"Baseline does not reproduce prior eval: {delta}")
        np.savez_compressed(output / f"{domain}.npz", allow_pickle=False, **arrays)
        prior.publish_results(output, report)
    if before_state != {
        k: prior.tensor_sha(v) for k, v in predictor.model.state_dict().items()
    }:
        raise ValueError("Model state mutated")
    report.update(status="completed", stage="finished", model_state_unchanged=True)
    prior.publish_results(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with prior.publish_run(args.output_dir) as report:
        run(args.output_dir, report)
