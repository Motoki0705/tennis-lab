"""Strict camera-local input validation and numerically stable MDN decoding."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_2d.model import Refiner2DModel
from src.tasks.base.model_io import BoundModelIO, ModelCall, bind_model_io
from src.tasks.base.model_io.tensors import TensorSpec


def _float_input(
    name: str, value: Tensor, shape: tuple[int | None, ...], device: torch.device
) -> None:
    TensorSpec(shape, frozenset({torch.float32})).validate(name, value)
    if value.device != device or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and on the candidates' device")


def _mask_input(
    name: str, value: Tensor, shape: tuple[int, ...], device: torch.device
) -> None:
    TensorSpec(shape, frozenset({torch.bool})).validate(name, value)
    if value.device != device:
        raise ValueError(f"{name} must share the candidates' device")


def _unit_interval(name: str, value: Tensor) -> None:
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must lie in [0,1]")


class Refiner2DAdapter:
    def __init__(self, config: Refiner2DConfig) -> None:
        self.config = config

    @property
    def model_type(self) -> type[nn.Module]:
        return Refiner2DModel

    def build_call(self, batch: Refiner2DInput) -> ModelCall:
        candidates = batch.candidates
        coords = candidates.coords
        TensorSpec((None, None, None, 2), frozenset({torch.float32})).validate(
            "candidate_coords", coords
        )
        b, t, n, _ = coords.shape
        if min(b, t, n) <= 0:
            raise ValueError("Batch, time and candidate axes must be nonempty")
        if (
            n != candidates.config.max_candidates
            or candidates.config.patch_size != self.config.patch_size
        ):
            raise ValueError("Candidate dimensions/config disagree with the refiner")
        p, device = self.config.patch_size, coords.device
        for name, value, shape in (
            ("candidate_coords", coords, (b, t, n, 2)),
            ("candidate_scores", candidates.scores, (b, t, n)),
            ("patches", candidates.patches, (b, t, n, p, p)),
        ):
            _float_input(name, value, shape, device)
            _unit_interval(name, value)
        _mask_input("candidate_valid", candidates.valid, (b, t, n), device)
        _mask_input("patch_valid", candidates.patch_valid, (b, t, n, p, p), device)
        if bool((candidates.patch_valid & ~candidates.valid[..., None, None]).any()):
            raise ValueError("Invalid candidates cannot have valid patch cells")
        if bool((candidates.patches[~candidates.patch_valid] != 0).any()):
            raise ValueError("Invalid patch cells must contain zero")
        if not torch.equal(
            candidates.patch_valid[..., p // 2, p // 2], candidates.valid
        ):
            raise ValueError("Every valid candidate requires a valid patch centre")
        if not torch.equal(
            candidates.patches[..., p // 2, p // 2][candidates.valid],
            candidates.scores[candidates.valid],
        ):
            raise ValueError("Patch centres must equal native candidate scores")
        _float_input("timestamps_seconds", batch.timestamps_seconds, (b, t), device)
        if bool(
            (batch.timestamps_seconds[:, 1:] <= batch.timestamps_seconds[:, :-1]).any()
        ):
            raise ValueError("Source frame timestamps must be strictly increasing")
        _float_input("pose_uv", batch.pose_uv, (b, t, None, 4, 2), device)
        people = batch.pose_uv.shape[2]
        _float_input(
            "pose_confidence", batch.pose_confidence, (b, t, people, 4), device
        )
        _unit_interval("pose_confidence", batch.pose_confidence)
        _mask_input("pose_valid", batch.pose_valid, (b, t, people, 4), device)
        c = self.config.court_keypoints
        _float_input("court_uv", batch.court_uv, (b, c, 2), device)
        _float_input("court_confidence", batch.court_confidence, (b, c), device)
        _unit_interval("court_confidence", batch.court_confidence)
        _mask_input("court_valid", batch.court_valid, (b, c), device)

        candidate_valid = candidates.valid & self.config.use_detector
        pose_valid = batch.pose_valid & self.config.use_pose
        court_valid = batch.court_valid & self.config.use_court
        features = torch.cat(
            (
                coords,
                candidates.scores.unsqueeze(-1),
                candidates.patches.flatten(-2),
                candidates.patch_valid.flatten(-2).float(),
            ),
            dim=-1,
        ).masked_fill(~candidate_valid.unsqueeze(-1), 0)
        pose = torch.cat((batch.pose_uv, batch.pose_confidence.unsqueeze(-1)), dim=-1)
        court = torch.cat(
            (batch.court_uv, batch.court_confidence.unsqueeze(-1)), dim=-1
        )
        return ModelCall(
            args=(
                features,
                candidate_valid,
                pose.masked_fill(~pose_valid.unsqueeze(-1), 0),
                pose_valid,
                court.masked_fill(~court_valid.unsqueeze(-1), 0),
                court_valid,
                batch.timestamps_seconds - batch.timestamps_seconds[:, :1],
            )
        )

    def decode_output(self, output: Tensor) -> BallGMM2D:
        TensorSpec(
            (None, None, self.config.components * 6 + 1),
            frozenset({torch.float16, torch.bfloat16, torch.float32}),
        ).validate("raw_refiner_output", output)
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Raw refiner output must be finite")
        # Nonlinear covariance transforms/linalg must stay float32 under AMP.
        raw = output.float()
        b, t, _ = raw.shape
        values = raw[..., :-1].reshape(b, t, self.config.components, 6)
        means = values[..., :2].sigmoid()
        std = (
            self.config.min_std
            + (self.config.max_std - self.config.min_std) * values[..., 2:4].sigmoid()
        )
        rho = self.config.max_correlation * values[..., 4].tanh()
        sx, sy = std.unbind(-1)
        row0 = torch.stack((sx, torch.zeros_like(sx)), dim=-1)
        row1 = torch.stack((rho * sy, sy * torch.sqrt(1 - rho.square())), dim=-1)
        return BallGMM2D(
            means=means,
            scale_tril=torch.stack((row0, row1), dim=-2),
            mixture_logits=values[..., 5],
            presence_logits=raw[..., -1],
        )


def build_ball_refiner_2d(
    config: Refiner2DConfig,
) -> BoundModelIO[Refiner2DInput, Tensor, BallGMM2D]:
    """Bind the architecture and input/output contract once."""
    return bind_model_io(Refiner2DModel(config), Refiner2DAdapter(config))
