"""Fixed-checkpoint validation diagnostics; neither fit calibration nor access test."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache, select_clips
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, validation_partition
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.evaluation.bootstrap import Statistic, grouped_intervals
from src.tasks.ball_refiner.evaluation.configuration import EvaluationConfig
from src.tasks.ball_refiner.evaluation.hdr import highest_density_regions
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import metric_rows, predict_clip
from src.tasks.base.training.compilation import compile_modules
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.device import resolve_device


def _summarize(
    rows: list[tuple[str, dict[str, NDArray[np.generic]]]], config: EvaluationConfig, *, observed: bool,
) -> dict[str, Any]:
    groups = [group for group, row in rows for _ in row["error_px"]]
    if not groups:
        raise ValueError("Validation stratum has no known positions")
    columns = {key: np.concatenate([row[key] for _, row in rows]) for key in rows[0][1]}
    error = columns["error_px"].astype(np.float64)
    quantities: dict[str, tuple[NDArray[np.float64], Statistic]] = {
        "mean_error_px": (error, "mean"), "median_error_px": (error, "median"), "p95_error_px": (error, "p95"),
        "recall_20px": ((error <= 20).astype(np.float64), "mean"),
        "position_nll_uv": (columns["nll_uv"].astype(np.float64), "mean"),
        "position_nll_px": (columns["nll_px"].astype(np.float64), "mean"),
        "mean_total_variance_px2": (columns["variance_px2"].astype(np.float64), "mean"),
    }
    for index, level in enumerate(config.levels):
        for name in ("coverage", "area_px2", "area_mc_standard_error_px2"):
            quantities[f"{name}_mass_{level:g}"] = (columns[name][:, index].astype(np.float64), "mean")
    if observed:
        detector = columns["unmasked_detector_error_px"].astype(np.float64)
        quantities.update({
            "detector_mean_error_px": (detector, "mean"), "detector_median_error_px": (detector, "median"),
            "detector_p95_error_px": (detector, "p95"), "detector_recall_20px": ((detector <= 20).astype(np.float64), "mean"),
            "paired_mean_error_delta_px": (error - detector, "mean"),
        })
    return {"frames": len(groups), "temporal_camera_groups": sorted(set(groups)),
            "metrics": grouped_intervals(quantities, groups, repetitions=config.bootstrap_repetitions,
                                         confidence=config.bootstrap_confidence, seed=config.seed)}


def run_evaluation(config_value: DictConfig) -> Path:
    config = EvaluationConfig.from_config(config_value)
    source = config.training_run
    input_files = [source / name for name in ("config.yaml", "data_manifest.json", "run_state.json", "best.json")]
    input_hashes = {str(path): dual_sha256(path) for path in input_files}
    state = json.loads((source / "run_state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Evaluation requires a completed training run")
    training = PilotConfig.from_config(OmegaConf.load(source / "config.yaml"))
    data = json.loads((source / "data_manifest.json").read_text())
    best = json.loads((source / "best.json").read_text())
    if best["checkpoint"] != f"epoch-{best['epoch']:03d}.pt":
        raise ValueError("Unexpected checkpoint filename")
    checkpoint_path = source / best["checkpoint"]
    input_hashes[str(checkpoint_path)] = dual_sha256(checkpoint_path)
    if input_hashes[str(checkpoint_path)] != best["checkpoint_sha256"]:
        raise ValueError("Best checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != "ball_refiner_2d_checkpoint.v1"
            or checkpoint["model_config"] != asdict(training.model)
            or checkpoint["data_manifest_sha256"] != input_hashes[str(source / "data_manifest.json")]
            or checkpoint["epoch"] != best["epoch"] or state["best_epoch"] != best["epoch"]
            or checkpoint["selection_nll_uv"] != best["selection_nll_uv"]):
        raise ValueError("Checkpoint config/data/selection identity mismatch")
    store = BallFrameStore(training.store)
    cache = EvidenceCache(training.evidence, store)
    cache_manifest = training.evidence / "manifest.json"
    input_hashes[str(cache_manifest)] = dual_sha256(cache_manifest)
    if input_hashes[str(cache_manifest)] != data["evidence_manifest_sha256"] or cache.manifest["store"]["sha256"] != data["store_sha256"]:
        raise ValueError("Evaluation inputs differ from the training evidence/store")
    records = select_clips(store, sources=("meiji",), splits=("val",))
    partition = validation_partition(records, training.partition_seed)
    if partition != data["validation"]:
        raise ValueError("Validation partition differs from the training manifest")
    selected = [record for record in records if record.clip_id in partition[config.partition]]
    if len({record.clip_id.rsplit("/", 1)[0] for record in selected}) < 2:
        raise ValueError("Diagnostics require at least two temporal camera groups for bootstrap")
    if any(record.frame_count < training.window_length for record in selected):
        raise ValueError("Evaluation requires full real windows; no short-clip padding")
    for path in (training.store, training.evidence):
        if config.output == path or config.output.is_relative_to(path) or path.is_relative_to(config.output):
            raise ValueError("Evaluation output must be separate from all input data")
    device = resolve_device(config.device)
    if config.output.exists() and any(path.name != "hydra" for path in config.output.iterdir()):
        raise FileExistsError(f"Evaluation output already contains a run: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output / "run_state.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "evaluating"}, stream)
    OmegaConf.save(OmegaConf.create(config.resolved), config.output / "config.yaml")
    prediction_dir = config.output / "predictions"
    prediction_dir.mkdir(exist_ok=False)
    pair = build_ball_refiner_2d(training.model)
    pair.model.load_state_dict(checkpoint["state_dict"], strict=True)
    pair.model.to(device)
    compile_modules({"refiner_2d": pair.model}, config.compilation)
    results: dict[str, list[tuple[str, dict[str, NDArray[np.generic]]]]] = {"observed": [], "evidence_gap": []}
    artifacts = []
    for record in selected:
        clip = LoadedClip(record, cache.load(record.clip_id), project_store_targets(store, record))
        gap = fixed_gap_mask(record.frame_count, clip_id=record.clip_id, block_length=training.window_length,
                             lengths=training.training.gap_lengths, seed=training.partition_seed)
        lengths = np.zeros(record.frame_count, dtype=np.int32)
        edges = np.diff(np.r_[False, gap, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            lengths[start:stop] = stop - start
        for condition in results:
            mask = gap if condition == "evidence_gap" else np.zeros_like(gap)
            prediction = predict_clip(pair, clip, training, device=device, gap=mask)
            score_mask = gap if condition == "evidence_gap" else np.ones_like(gap)
            located = clip.targets.position_valid & score_mask
            if not located.any():
                raise ValueError(f"No located targets in {record.clip_id}/{condition}")
            distribution = BallGMM2D(**{
                field.name: getattr(prediction, field.name)[:, located].to(device) for field in fields(prediction)
            })
            seed = int.from_bytes(hashlib.sha256(f"{config.seed}:{record.clip_id}:{condition}".encode()).digest()[:8], "little") % (2**63 - 1)
            hdr = highest_density_regions(distribution, torch.from_numpy(clip.targets.uv[located])[None].to(device),
                                          levels=config.levels, samples=config.samples, seed=seed, chunk_size=config.chunk_size)
            row = metric_rows(prediction, clip, score_mask)
            # Presence rows may have a different mask; only located frame quantities are bootstrapped.
            row = {key: row[key] for key in ("nll_uv", "nll_px", "error_px", "variance_px2")}
            scale = np.array([record.source_width - 1, record.source_height - 1], dtype=np.float64)
            row.update({
                "coverage": hdr.covered[0].numpy(), "area_px2": hdr.area_uv2[0].numpy() * scale.prod(),
                "area_mc_standard_error_px2": hdr.area_mc_standard_error_uv2[0].numpy() * scale.prod(),
                "unmasked_detector_error_px": np.linalg.norm((clip.evidence.argmax_uv[located] - clip.targets.uv[located]) * scale, axis=-1),
                "gap_length": lengths[located],
            })
            results[condition].append((record.clip_id.rsplit("/", 1)[0], row))
            output = prediction_dir / f"clip-{record.index:05d}-{condition}.npz"
            with output.open("xb") as stream:
                np.savez_compressed(stream, **{field.name: getattr(prediction, field.name)[0].numpy() for field in fields(prediction)},
                                    **row, frame_index=clip.evidence.frame_index, pts=clip.evidence.pts,
                                    timestamps_seconds=clip.evidence.timestamps_seconds, gap_mask=mask,
                                    target_uv=clip.targets.uv, position_valid=clip.targets.position_valid,
                                    scored_frame_index=clip.evidence.frame_index[located],
                                    hdr_log_threshold_uv=hdr.log_threshold[0].numpy())
            artifacts.append({"clip_id": record.clip_id, "condition": condition, "frames": record.frame_count,
                              "scored_frames": int(located.sum()), "source_size_wh": [record.source_width, record.source_height],
                              "path": str(output.relative_to(config.output)), "sha256": dual_sha256(output), "mc_seed": seed})
            write_json_atomic(config.output / "progress.json", {"completed": artifacts})
            print(json.dumps(artifacts[-1]), flush=True)
    report = {condition: _summarize(rows, config, observed=condition == "observed") for condition, rows in results.items()}
    # Same source frames under intact evidence vs missing evidence, stratified by gap duration.
    for condition, rows in results.items():
        for length in training.training.gap_lengths:
            subset = [(group, {key: value[row["gap_length"] == length] for key, value in row.items()}) for group, row in rows]
            report[f"{condition}_at_gap_length_{length}"] = _summarize(subset, config, observed=condition == "observed")
    if any(dual_sha256(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError("Evaluation input changed during execution")
    write_json_atomic(config.output / "manifest.json", {
        "schema": "ball_refiner_validation_diagnostics.v1", "partition": config.partition,
        "clip_ids": partition[config.partition], "checkpoint": best, "input_sha256": input_hashes,
        "artifacts": artifacts, "hdr_levels": config.levels,
        "hdr": "conditional GMM on R2; float64 MC; independent fit/area draws; area SE excludes threshold uncertainty",
        "bootstrap": "whole meiji/video/clip with every camera; frame-weighted percentile intervals",
        "comparison": "unthresholded detector argmax on observed only; no detector density or gap baseline",
        "scope": "validation diagnostic; no calibration fitting, no test, no RGB occlusion, no real amodal GT",
    })
    write_json_atomic(config.output / "metrics.json", report)
    write_json_atomic(config.output / "run_state.json", {"status": "complete", "partition": config.partition,
                                                       "camera_clips": len(selected), "prediction_files": len(artifacts)})
    return config.output
