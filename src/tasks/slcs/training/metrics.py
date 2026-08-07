"""Evaluation metrics for SLCS.

Player metrics mirror :class:`src.tasks.plcs.training.metrics.PLCSMetrics`
(position error in meters, angular error in degrees, fixed-threshold
accuracies) and ball metrics mirror
:class:`src.tasks.blcs.training.metrics.BLCSMetrics` (L2 error in meters,
fixed-threshold accuracies), with ``player_`` / ``ball_`` prefixes, so SLCS
results are directly comparable with the existing single-task baselines.

Additional confidence diagnostics: mean predicted Laplace scale ``b`` (in
meters for positions, degrees for yaw) and the Pearson correlation between
predicted ``b`` and the actual per-frame error — a quick calibration signal
(detailed reliability curves live in the analysis script).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.tasks.slcs.model_io import SLCSDecodedOutput, SLCSTrainingTargets
from src.utils.geometry.angles import angular_error
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

_POSITION_THRESHOLDS_M = (0.3, 0.5, 1.0, 2.0)
_ANGLE_THRESHOLDS_DEG = (10.0, 15.0, 30.0)


def _pearson(x: Tensor, y: Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0.0:
        return 0.0
    return float((x * y).sum().item() / denom.item())


class SLCSMetrics:
    """Accumulate masked player/ball errors over batches."""

    def __init__(self) -> None:
        self._scale = torch.tensor(list(COURT_COORD_SCALE_XYZ), dtype=torch.float32)
        # Mean position scale for converting normalized Laplace b to meters.
        self._scale_mean = float(sum(COURT_COORD_SCALE_XYZ) / 3.0)
        self.reset()

    def reset(self) -> None:
        self._player_pos_errors: list[Tensor] = []
        self._player_ang_errors: list[Tensor] = []
        self._ball_pos_errors: list[Tensor] = []
        self._player_pos_b: list[Tensor] = []
        self._player_rot_b: list[Tensor] = []
        self._ball_pos_b: list[Tensor] = []

    def update(
        self,
        outputs: SLCSDecodedOutput,
        targets: SLCSTrainingTargets,
    ) -> dict[str, float]:
        """Accumulate one batch; returns current-batch means for logging."""
        player_mask = targets.player_mask
        ball_mask = targets.ball_mask

        scale = self._scale.to(outputs.player_position.device)

        result: dict[str, float] = {}
        if bool(player_mask.any()):
            pred_pos = outputs.player_position[player_mask] * scale
            target_pos = targets.target_player_position[player_mask] * scale
            pos_err = (pred_pos - target_pos).norm(dim=-1)
            ang_err_deg = (
                angular_error(
                    outputs.player_rotation[player_mask],
                    targets.target_player_rotation[player_mask],
                )
                * 180.0
                / math.pi
            )
            pos_b = (
                outputs.player_position_log_b[player_mask].exp() * self._scale_mean
            )
            rot_b = (
                outputs.player_rotation_log_b[player_mask].exp() * 180.0 / math.pi
            )
            self._player_pos_errors.append(pos_err.detach().cpu())
            self._player_ang_errors.append(ang_err_deg.detach().cpu())
            self._player_pos_b.append(pos_b.detach().cpu())
            self._player_rot_b.append(rot_b.detach().cpu())
            result["player_position_error_m"] = float(pos_err.mean().item())
            result["player_angular_error_deg"] = float(ang_err_deg.mean().item())
        if bool(ball_mask.any()):
            pred_ball = outputs.ball_position[ball_mask] * scale
            target_ball = targets.target_ball_position[ball_mask] * scale
            ball_err = (pred_ball - target_ball).norm(dim=-1)
            ball_b = outputs.ball_position_log_b[ball_mask].exp() * self._scale_mean
            self._ball_pos_errors.append(ball_err.detach().cpu())
            self._ball_pos_b.append(ball_b.detach().cpu())
            result["ball_position_error_m"] = float(ball_err.mean().item())
        return result

    def compute(self) -> dict[str, float]:
        """Aggregate metrics over all accumulated batches."""
        out: dict[str, float] = {}

        if self._player_pos_errors:
            pos = torch.cat(self._player_pos_errors)
            ang = torch.cat(self._player_ang_errors)
            pos_b = torch.cat(self._player_pos_b)
            rot_b = torch.cat(self._player_rot_b)
            out["player_position_error_m"] = float(pos.mean().item())
            out["player_position_error_median_m"] = float(pos.median().item())
            out["player_angular_error_deg"] = float(ang.mean().item())
            out["player_angular_error_median_deg"] = float(ang.median().item())
            for thr in _POSITION_THRESHOLDS_M:
                out[f"player_position_accuracy_{thr}m"] = float(
                    (pos <= thr).float().mean().item()
                )
            for thr in _ANGLE_THRESHOLDS_DEG:
                out[f"player_angle_accuracy_{int(thr)}deg"] = float(
                    (ang <= thr).float().mean().item()
                )
            out["player_position_pred_b_m"] = float(pos_b.mean().item())
            out["player_rotation_pred_b_deg"] = float(rot_b.mean().item())
            out["player_position_conf_error_corr"] = _pearson(pos_b, pos)
            out["player_rotation_conf_error_corr"] = _pearson(rot_b, ang)

        if self._ball_pos_errors:
            ball = torch.cat(self._ball_pos_errors)
            ball_b = torch.cat(self._ball_pos_b)
            out["ball_position_error_m"] = float(ball.mean().item())
            out["ball_position_error_median_m"] = float(ball.median().item())
            for thr in _POSITION_THRESHOLDS_M:
                out[f"ball_position_accuracy_{thr}m"] = float(
                    (ball <= thr).float().mean().item()
                )
            out["ball_position_pred_b_m"] = float(ball_b.mean().item())
            out["ball_position_conf_error_corr"] = _pearson(ball_b, ball)

        return out


__all__ = ["SLCSMetrics"]
