"""Label-free input slicing, person-axis collation and device transfer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields

import torch

from src.tasks.ball_detection.model_io.contracts import BallCandidates
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput

CANDIDATE_FIELDS = ("coords", "scores", "valid", "cells", "patches", "patch_valid")
TEMPORAL_FIELDS = ("timestamps_seconds", "pose_uv", "pose_confidence", "pose_valid")
STATIC_FIELDS = ("court_uv", "court_confidence", "court_valid")


def input_to(inputs: Refiner2DInput, device: torch.device) -> Refiner2DInput:
    candidates = BallCandidates(
        **{name: getattr(inputs.candidates, name).to(device) for name in CANDIDATE_FIELDS},
        config=inputs.candidates.config,
    )
    return Refiner2DInput(candidates=candidates, **{
        field.name: getattr(inputs, field.name).to(device)
        for field in fields(inputs) if field.name != "candidates"
    })


def slice_input(inputs: Refiner2DInput, start: int, stop: int) -> Refiner2DInput:
    """Copy one camera's real frames; the static court never gains a time axis."""
    times = inputs.timestamps_seconds
    if times.ndim != 2 or times.shape[0] != 1 or not 0 <= start < stop <= times.shape[1]:
        raise ValueError("Input slice requires one camera and a nonempty real frame range")
    candidates = BallCandidates(**{
        name: getattr(inputs.candidates, name)[:, start:stop].clone() for name in CANDIDATE_FIELDS
    }, config=inputs.candidates.config)
    return Refiner2DInput(candidates=candidates, **{
        name: getattr(inputs, name)[:, start:stop].clone() for name in TEMPORAL_FIELDS
    }, **{name: getattr(inputs, name).clone() for name in STATIC_FIELDS})


def detector_only_input(
    evidence: ClipEvidence, start: int, length: int, config: Refiner2DConfig,
) -> Refiner2DInput:
    """Construct the explicitly configured ablation; never invent missing context."""
    if config.use_pose or config.use_court or not config.use_detector:
        raise ValueError("Detector-only input requires use_detector=true, use_pose/use_court=false")
    if not 0 <= start < start + length <= len(evidence.frame_index):
        raise ValueError("Window must contain only real source frames")
    if config.patch_size != evidence.candidates.config.patch_size:
        raise ValueError("Model and evidence patch sizes differ")
    candidates = BallCandidates(**{
        name: getattr(evidence.candidates, name)[:, start:start + length].clone()
        for name in CANDIDATE_FIELDS
    }, config=evidence.candidates.config)
    return Refiner2DInput(
        candidates=candidates,
        timestamps_seconds=torch.from_numpy(evidence.timestamps_seconds[start:start + length].copy())[None],
        pose_uv=torch.zeros(1, length, 0, 4, 2), pose_confidence=torch.zeros(1, length, 0, 4),
        pose_valid=torch.zeros(1, length, 0, 4, dtype=torch.bool),
        court_uv=torch.zeros(1, config.court_keypoints, 2), court_confidence=torch.zeros(1, config.court_keypoints),
        court_valid=torch.zeros(1, config.court_keypoints, dtype=torch.bool),
    )


def collate_inputs(inputs: Sequence[Refiner2DInput]) -> Refiner2DInput:
    """Pad only the unordered person axis; temporal and candidate axes must match."""
    if not inputs:
        raise ValueError("Cannot collate an empty input batch")
    length = inputs[0].timestamps_seconds.shape[1]
    config = inputs[0].candidates.config
    if any(x.timestamps_seconds.shape != (1, length) or x.candidates.config != config for x in inputs):
        raise ValueError("Collation requires matching real time windows and candidate settings")
    people = max(x.pose_uv.shape[2] for x in inputs)
    pose: dict[str, torch.Tensor] = {}
    for name in ("pose_uv", "pose_confidence", "pose_valid"):
        values = []
        for item in inputs:
            value = getattr(item, name)
            padded = value.new_zeros((1, length, people, *value.shape[3:]))
            padded[:, :, :value.shape[2]] = value
            values.append(padded)
        pose[name] = torch.cat(values)
    candidates = BallCandidates(**{
        name: torch.cat([getattr(x.candidates, name) for x in inputs]) for name in CANDIDATE_FIELDS
    }, config=config)
    return Refiner2DInput(candidates=candidates, **pose, **{
        name: torch.cat([getattr(x, name) for x in inputs])
        for name in ("timestamps_seconds", *STATIC_FIELDS)
    })
