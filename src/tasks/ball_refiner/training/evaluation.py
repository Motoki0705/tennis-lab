"""Pilot validation: one prediction per source frame and explicit evidence gaps."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, mask_detector_evidence
from src.tasks.ball_refiner.data.windows import (
    LoadedClip,
    collate_windows,
    detector_only_window,
    window_owners,
    window_starts,
)
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput
from src.tasks.ball_refiner.refiner_2d.distribution import (
    BallGMM2D,
    conditional_log_density,
)
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.base.model_io import BoundModelIO

RefinerPair: TypeAlias = BoundModelIO[Refiner2DInput, torch.Tensor, BallGMM2D]


def predict_clip(
    pair: RefinerPair, clip: LoadedClip, config: PilotConfig, *, device: torch.device,
    gap: NDArray[np.bool_],
) -> BallGMM2D:
    """Copy the complete GMM from one selected window; never average components."""
    frames = clip.record.frame_count
    if gap.dtype != np.bool_ or gap.shape != (frames,):
        raise ValueError("Global gap mask must address every source frame")
    length = config.window_length
    starts = window_starts(frames, length, config.stride)
    owners = window_owners(frames, starts, length)
    collected: dict[str, torch.Tensor] = {}
    pair.model.eval()
    for offset in range(0, len(starts), config.training.batch_size):
        selected = starts[offset:offset + config.training.batch_size]
        batch = collate_windows([detector_only_window(clip, start, length, config.model) for start in selected])
        gap_batch = torch.from_numpy(np.stack([gap[start:start + length] for start in selected]))
        inputs = mask_detector_evidence(batch.inputs, gap_batch)
        batch = type(batch)(inputs, batch.target).to(device)
        with torch.no_grad():
            prediction = pair.run(batch.inputs)
        for field in fields(prediction):
            values = getattr(prediction, field.name).cpu()
            if field.name not in collected:
                collected[field.name] = torch.empty((1, frames, *values.shape[2:]), dtype=values.dtype)
            for row, start in enumerate(selected):
                indices = np.flatnonzero(owners[start:start + length] == offset + row)
                collected[field.name][0, start + indices] = values[row, indices]
    return BallGMM2D(**collected)


def metric_rows(
    prediction: BallGMM2D, clip: LoadedClip, selected: NDArray[np.bool_],
) -> dict[str, NDArray[np.generic]]:
    """Return frame-level quantities so aggregation cannot average batch means."""
    target = clip.targets
    located = torch.from_numpy(selected & target.position_valid)
    known = torch.from_numpy(selected & target.presence_valid)
    uv = torch.from_numpy(target.uv)[located]
    means, logits = prediction.means[0, located], prediction.mixture_logits[0, located]
    tril = prediction.scale_tril[0, located]
    nll = -conditional_log_density(uv, means, tril, logits)
    scale = torch.tensor([clip.record.source_width - 1, clip.record.source_height - 1], dtype=torch.float32)
    selected_mean = means[torch.arange(len(means)), logits.argmax(-1)]
    errors = torch.linalg.vector_norm((selected_mean - uv) * scale, dim=-1)
    presence = torch.from_numpy(target.presence)[known].float()
    presence_logits = prediction.presence_logits[0, known]
    bce = F.binary_cross_entropy_with_logits(presence_logits, presence, reduction="none")
    weights = logits.softmax(-1)
    expected = (means * weights[..., None]).sum(-2)
    covariance = tril @ tril.transpose(-1, -2)
    variance = ((covariance.diagonal(dim1=-2, dim2=-1) + (means - expected[:, None]).square()) * weights[..., None]).sum(-2)
    return {
        "nll_uv": nll.numpy(), "nll_px": (nll + scale.log().sum()).numpy(),
        "error_px": errors.numpy(), "presence_bce": bce.numpy(),
        "presence_brier": (presence_logits.sigmoid() - presence).square().numpy(),
        "presence_label": presence.numpy(),
        "variance_px2": (variance * scale.square()).sum(-1).numpy(),
    }


def summarize_rows(rows: list[dict[str, NDArray[np.generic]]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Validation selection cannot be empty")
    joined = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    count, known = len(joined["nll_uv"]), len(joined["presence_label"])
    if count == 0 or known == 0:
        raise ValueError("Validation stratum needs known positions and presence")
    return {
        "position_frames": count, "presence_frames": known,
        "positive_frames": int(joined["presence_label"].sum()),
        "negative_frames": int(known - joined["presence_label"].sum()),
        "position_nll_uv": float(joined["nll_uv"].mean(dtype=np.float64)),
        "position_nll_px": float(joined["nll_px"].mean(dtype=np.float64)),
        "mean_error_px": float(joined["error_px"].mean(dtype=np.float64)),
        "median_error_px": float(np.median(joined["error_px"])),
        "p95_error_px": float(np.quantile(joined["error_px"], .95)),
        "recall_20px": float((joined["error_px"] <= 20).mean()),
        "presence_bce": float(joined["presence_bce"].mean(dtype=np.float64)),
        "presence_brier": float(joined["presence_brier"].mean(dtype=np.float64)),
        "mean_total_variance_px2": float(joined["variance_px2"].mean(dtype=np.float64)),
    }


def evaluate_selection(
    pair: RefinerPair, clips: tuple[LoadedClip, ...], config: PilotConfig, device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, NDArray[np.generic]]]]:
    rows: dict[str, list[dict[str, NDArray[np.generic]]]] = {"observed": [], "evidence_gap": []}
    predictions: dict[str, dict[str, NDArray[np.generic]]] = {}
    per_clip: list[dict[str, Any]] = []
    for clip in clips:
        gap = fixed_gap_mask(clip.record.frame_count, clip_id=clip.record.clip_id, block_length=config.window_length,
                             lengths=config.training.gap_lengths, seed=config.partition_seed)
        for condition in rows:
            mask = gap if condition == "evidence_gap" else np.zeros_like(gap)
            prediction = predict_clip(pair, clip, config, device=device, gap=mask)
            selected = gap if condition == "evidence_gap" else np.ones_like(gap)
            row = metric_rows(prediction, clip, selected)
            rows[condition].append(row)
            per_clip.append({"clip_id": clip.record.clip_id, "condition": condition,
                             "position_frames": len(row["nll_uv"]), "presence_frames": len(row["presence_label"])})
            key = f"clip-{clip.record.index:05d}-{condition}"
            predictions[key] = {
                **{f.name: getattr(prediction, f.name)[0].numpy() for f in fields(prediction)},
                "frame_index": clip.evidence.frame_index, "pts": clip.evidence.pts,
                "timestamps_seconds": clip.evidence.timestamps_seconds,
                "gap_mask": mask, "target_uv": clip.targets.uv,
                "position_valid": clip.targets.position_valid, "presence_valid": clip.targets.presence_valid,
                "presence": clip.targets.presence,
            }
    report = {condition: summarize_rows(value) for condition, value in rows.items()}
    selection = .5 * (report["observed"]["position_nll_uv"] + report["evidence_gap"]["position_nll_uv"])
    return {"selection_nll_uv": selection, **report, "per_clip": per_clip}, predictions
