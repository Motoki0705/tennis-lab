"""CPU audit of this historical pilot; run from the captured checkout with PYTHONPATH=."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache, select_clips
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import (
    metric_rows,
    predict_clip,
    summarize_rows,
)
from src.utils.checksum import dual_sha256


def audit(directory: Path) -> dict[str, Any]:
    torch.set_num_threads(4)
    config = PilotConfig.from_config(OmegaConf.load(directory / "config.yaml"))
    manifest = json.loads((directory / "data_manifest.json").read_text())
    manifest_hash = dual_sha256(directory / "data_manifest.json")
    state = json.loads((directory / "run_state.json").read_text())
    best = json.loads((directory / "best.json").read_text())
    curve = [json.loads(line) for line in (directory / "learning_curve.jsonl").read_text().splitlines()]
    assert state["status"] == "complete"
    assert state["steps"] == min(config.training.epochs * config.training.steps_per_epoch, config.training.max_steps)
    assert best["epoch"] == min(curve, key=lambda row: row["selection_nll_uv"])["epoch"]
    assert dual_sha256(directory / best["checkpoint"]) == best["checkpoint_sha256"]
    torch.manual_seed(config.seed)
    pair = build_ball_refiner_2d(config.model)
    initial = {key: value.clone() for key, value in pair.model.state_dict().items()}
    checkpoints = []
    for row in curve:
        path = directory / f"epoch-{row['epoch']:03d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == row["epoch"]
        assert checkpoint["step"] == row["step"]
        assert checkpoint["data_manifest_sha256"] == manifest_hash
        assert checkpoint["selection_nll_uv"] == row["selection_nll_uv"]
        optimizer_states = list(checkpoint["optimizer"]["state"].values())
        optimizer_steps = sorted({int(value["step"]) for value in optimizer_states})
        assert optimizer_steps == [row["step"]]
        assert all(bool(torch.isfinite(value["exp_avg"]).all()) for value in optimizer_states)
        assert all(bool(torch.isfinite(value["exp_avg_sq"]).all()) for value in optimizer_states)
        assert any(bool((value["exp_avg"] != 0).any()) for value in optimizer_states)
        parameters = checkpoint["state_dict"]
        assert set(parameters) == set(initial)
        assert all(bool(torch.isfinite(value).all()) for value in parameters.values())
        changed = sum(not torch.equal(parameters[key], initial[key]) for key in parameters)
        assert changed > 0
        checkpoints.append({
            "epoch": row["epoch"], "step": row["step"], "sha256": dual_sha256(path),
            "optimizer_state_count": len(optimizer_states), "optimizer_steps": optimizer_steps,
            "changed_state_tensors_from_seeded_initialization": changed,
            "total_state_tensors": len(parameters),
        })
    checkpoint = torch.load(directory / best["checkpoint"], map_location="cpu", weights_only=True)
    pair.model.load_state_dict(checkpoint["state_dict"], strict=True)
    pair.model.eval()
    store = BallFrameStore(config.store)
    cache = EvidenceCache(config.evidence, store)
    assert dual_sha256(config.evidence / "manifest.json") == manifest["evidence_manifest_sha256"]
    records = {record.clip_id: record for record in select_clips(store, splits=("val",), sources=("meiji",))}
    selection = manifest["validation"]["selection"]
    assert not set(selection) & set(manifest["validation"]["calibration"])
    replay_id = min(selection, key=lambda key: records[key].frame_count)
    summaries: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    replay: dict[str, Any] = {}
    detector_errors = []
    detector_scores = []
    for condition in ("observed", "evidence_gap"):
        rows = []
        for clip_id in selection:
            record = records[clip_id]
            clip = LoadedClip(record, cache.load(clip_id), project_store_targets(store, record))
            path = directory / best["predictions"] / f"clip-{record.index:05d}-{condition}.npz"
            hashes[path.name] = dual_sha256(path)
            with np.load(path, allow_pickle=False) as file:
                arrays = {key: file[key] for key in file.files}
            for key in ("frame_index", "pts", "timestamps_seconds"):
                np.testing.assert_array_equal(arrays[key], getattr(clip.evidence, key))
            for stored, source in (("target_uv", "uv"), ("position_valid", "position_valid"),
                                   ("presence_valid", "presence_valid"), ("presence", "presence")):
                np.testing.assert_array_equal(arrays[stored], getattr(clip.targets, source))
            gap = fixed_gap_mask(record.frame_count, clip_id=clip_id, block_length=config.window_length,
                                 lengths=config.training.gap_lengths, seed=config.partition_seed)
            mask = gap if condition == "evidence_gap" else np.zeros_like(gap)
            np.testing.assert_array_equal(arrays["gap_mask"], mask)
            prediction = BallGMM2D(**{
                field.name: torch.from_numpy(arrays[field.name])[None] for field in fields(BallGMM2D)
            })
            selected = gap if condition == "evidence_gap" else np.ones_like(gap)
            rows.append(metric_rows(prediction, clip, selected))
            if condition == "observed":
                valid = clip.targets.position_valid
                scale = np.array([record.source_width - 1, record.source_height - 1])
                detector_errors.append(np.linalg.norm((clip.evidence.argmax_uv[valid] - clip.targets.uv[valid]) * scale, axis=-1))
                detector_scores.append(clip.evidence.argmax_score[valid])
            if clip_id == replay_id:
                restored = predict_clip(pair, clip, config, device=torch.device("cpu"), gap=mask)
                differences = {}
                for field in fields(prediction):
                    actual, expected = getattr(restored, field.name), getattr(prediction, field.name)
                    # Compiled CUDA vs eager CPU: logits near zero need an absolute tolerance.
                    tolerance = 3e-4 if field.name.endswith("logits") else 3e-5
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=3e-4)
                    differences[field.name] = float((actual - expected).abs().max())
                source_scale = torch.tensor([record.source_width - 1, record.source_height - 1])
                mean_difference_px = float(((restored.means - prediction.means).abs() * source_scale).max())
                assert mean_difference_px < .5
                replay[condition] = {"clip_id": clip_id, "frames": record.frame_count,
                                     "max_abs_difference": differences, "max_mean_coordinate_difference_px": mean_difference_px,
                                     "means_scale_atol": 3e-5, "logits_atol": 3e-4, "rtol": 3e-4}
        summaries[condition] = summarize_rows(rows)
    saved_metrics = json.loads((directory / "metrics.json").read_text())
    for condition, summary in summaries.items():
        for key, value in summary.items():
            np.testing.assert_allclose(value, saved_metrics[condition][key], rtol=1e-7, atol=1e-7)
    errors, scores = np.concatenate(detector_errors), np.concatenate(detector_scores)
    return {
        "scope": "selection validation; not calibration/test; evidence gaps, not RGB occlusion",
        "run_state": state, "best": best, "model_parameters": sum(p.numel() for p in pair.model.parameters()),
        "data_manifest_sha256": manifest_hash, "evidence_manifest_sha256": manifest["evidence_manifest_sha256"],
        "checkpoints": checkpoints, "cpu_checkpoint_replay": replay,
        "prediction_sha256": hashes, "recomputed_metrics": summaries,
        "detector_argmax_observed_same_frames": {
            "frames": len(errors), "mean_error_px": float(errors.mean()),
            "median_error_px": float(np.median(errors)), "p95_error_px": float(np.quantile(errors, .95)),
            "recall_20px": float((errors <= 20).mean()), "mean_uncalibrated_score": float(scores.mean()),
            "selection": "all known observed positions; no detector score/refiner presence threshold",
        },
        "limitations": ["no HDR coverage or bootstrap", "no dense detector likelihood", "no real amodal labels",
                        "no context ablation", "no test split", "no raw media/JPEG/annotation rehash"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), ensure_ascii=False, indent=2, allow_nan=False))
