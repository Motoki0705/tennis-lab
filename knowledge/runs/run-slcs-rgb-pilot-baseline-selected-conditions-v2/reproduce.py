"""Validation-only selection followed by paired CPU FP32 input-condition evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.slcs.configuration import SLCSEvaluationConfig
from src.tasks.slcs.evaluation.comparison import compare_conditions, save_comparison
from src.tasks.slcs.evaluation.evaluate import (
    evaluate_split,
    evaluation_context,
    save_evaluation,
)
from src.tasks.slcs.inference.predictor import SLCSPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["baseline", "augmented"])
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    root = Path("/home/kamimura/projects/tennis-lab")
    output = root / "outputs"
    training = output / f"slcs/train/real_rgb_pilot_{args.model}/s42-002"
    analysis = output / f"slcs/analyze/real_rgb_pilot_{args.model}/s42-002"
    cfg = OmegaConf.load(training / "config.yaml")
    last = torch.load(training / "logs/version_0/checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 59
    callbacks = [c for c in last["callbacks"].values() if c.get("monitor") == "val/scene_position_error_m_epoch"]
    assert len(callbacks) == 1
    candidates = []
    for name, score in callbacks[0]["best_k_models"].items():
        path = Path(name)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append({"path": name, "epoch_zero_based": checkpoint["epoch"], "validation_scene_position_error_m": float(score), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    chosen = min(candidates, key=lambda c: c["validation_scene_position_error_m"])
    destination = root / f"ckpt/slcs/real-rgb-pilot-{args.model}-e60-v2.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == chosen["sha256"]
    else:
        shutil.copy2(chosen["path"], destination)
    receipt = {"selection": "minimum validation monitor among retained top-k; no test selection", "monitor": callbacks[0]["monitor"], "candidates": candidates, "selected": chosen, "destination": str(destination), "last_epoch_zero_based": last["epoch"], "terminal_test": "Training terminal test uses last model; separate from this selected-checkpoint evaluation"}
    (analysis / "selection.json").write_text(json.dumps(receipt, indent=2))
    destination.with_suffix(".metadata.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"model": args.model, "selected": chosen}), flush=True)
    if args.select_only:
        return
    cfg.paths.checkpoint_root = str(root / "ckpt")
    cfg.data.quality.min_window_label_ratio = 0.5
    evaluation = OmegaConf.create({"paths": OmegaConf.to_container(cfg.paths, resolve=True), "data": OmegaConf.to_container(cfg.data, resolve=True), "evaluate": {"checkpoint": f"slcs/{destination.name}", "split": "val", "device": "cpu", "batch_size": 4, "input_mode": "full", "checkpoint_strict": True, "checkpoint_weights_only": False, "output_dir": "slcs/evaluate/placeholder/s42-002"}})
    runtime = SLCSEvaluationConfig.from_config(evaluation)
    predictor = SLCSPredictor.load_from_checkpoint(runtime.checkpoint, resolver=runtime.resolver, device="cpu", strict=True, weights_only=False)
    predictor.model.float()
    for split in ("val", "test"):
        bundles = {}
        for condition in ("full", "no_rgb", "detector_gap", "rgb_only"):
            directory = output / f"slcs/evaluate/real_rgb_pilot_{args.model}_{split}_{condition}/s42-002"
            report, arrays = evaluate_split(predictor, dataset_root=runtime.data.dataset_root, split_file=runtime.data.split_file, split=split, data_config=runtime.data.pipeline, batch_size=4, input_mode=condition)
            context = evaluation_context(destination, input_mode=condition)
            context.update({"split": split, "dataset_root": str(runtime.data.dataset_root), "data_config": OmegaConf.to_container(cfg.data, resolve=True), "precision": "cpu float32", "selected_epoch_zero_based": chosen["epoch_zero_based"]})
            save_evaluation(directory, report, arrays, context=context)
            bundles[condition] = directory
            print(json.dumps({"model": args.model, "split": split, "condition": condition, "metrics": report}), flush=True)
        with np.load(bundles["full"] / "eval_arrays.npz") as archive:
            domains = {str(video): "meiji" if str(video).startswith("video_") else "broadcast" for video in archive["video_ids"]}
        comparison = compare_conditions(bundles, domains)
        comparison["domain_rule"] = "explicit video_ prefix = meiji; all others = broadcast"
        save_comparison(comparison, analysis / split)


if __name__ == "__main__":
    main()
