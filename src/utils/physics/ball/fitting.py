"""Fit the shared force model to trajectories and measure what it cannot explain.

For every rally the fit shares one field (k_drag, k_magnus, horizontal wind)
and gives every flight segment its own initial position, velocity and spin.
Gravity is fixed.  The residual between the trajectory and the best fitted
re-simulation is the part of the motion no consistent physics produces; ground
truth reaches the optimizer's precision.

The least-squares problem is solved by Levenberg-Marquardt per rally.  Each
segment's residual depends on its 9 own parameters and the 4 field parameters,
so forward-mode Jacobians of all segments are computed in one batched pass and
assembled into one small dense normal system per rally.

The field is not convex: from a single start, or when the field moves before
the segment spins are fitted, a rally can settle in a wrong wind/drag basin.
Every rally is therefore fitted from several field starts in one batch, first
with the field held fixed, and keeps its best result.

Spin enters free flight only through ``k_magnus * (omega x v)``: the fit
identifies that product, not ``k_magnus`` and spin separately, and spin parallel
to the velocity is unobservable.  The fitted field is a diagnostic, not a
parameter estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.func import jacfwd, vmap

from src.utils.physics.ball.dynamics import BallField
from src.utils.physics.ball.integrate import integrate_flight

_SEGMENT = 9  # position, velocity, spin / 100
_FIELD = 4  # log k_drag, log k_magnus, wind_x, wind_y
_SPIN_UNIT = 100.0


@dataclass(frozen=True)
class FitSettings:
    gravity: float
    dt: float
    substeps: int
    # Segment-only iterations at the fixed start field, then joint iterations.
    segment_iterations: int
    iterations: int
    minimum_segment_frames: int
    # Initial (k_drag, k_magnus, wind_x, wind_y); every rally keeps its best
    # fit over the starts.
    field_starts: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self) -> None:
        if min(self.substeps, self.iterations) < 1 or self.minimum_segment_frames < 3:
            raise ValueError("Require positive iterations/substeps and >=3 frames")
        if self.segment_iterations < 0:
            raise ValueError("segment_iterations must be nonnegative")
        if not (self.gravity > 0 and self.dt > 0):
            raise ValueError("Require positive gravity and dt")
        if not self.field_starts or any(
            len(start) != 4 or min(start[:2]) <= 0 for start in self.field_starts
        ):
            raise ValueError("Require (k_drag>0, k_magnus>0, wind_x, wind_y) starts")


@dataclass(frozen=True)
class FlightFit:
    """Per-frame residual ``(T,3)`` (NaN outside fitted segments) and field."""

    residual: NDArray[np.float64]
    k_drag: float
    k_magnus: float
    wind_xy: NDArray[np.float64]

    @property
    def fitted(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = np.isfinite(self.residual[:, 0])
        return result


def fit_flight_physics(
    trajectories: list[NDArray[np.floating]],
    segments: list[list[tuple[int, int]]],
    settings: FitSettings,
    device: torch.device,
) -> list[FlightFit]:
    """Fit every rally; ``segments`` lists ``[start, end)`` frame spans.

    The problem is not convex in the field: from one start a rally can settle
    in a wrong (k_drag, k_magnus) basin, so all starts run as one batch.
    """
    if len(trajectories) != len(segments) or not trajectories:
        raise ValueError("Require one segment list per trajectory")
    starts = len(settings.field_starts)
    fits, losses = _fit_from(
        trajectories * starts,
        segments * starts,
        [start for start in settings.field_starts for _ in trajectories],
        settings,
        device,
    )
    rallies = len(trajectories)
    best = losses.reshape(starts, rallies).argmin(axis=0)
    return [fits[int(start) * rallies + rally] for rally, start in enumerate(best)]


def _fit_from(
    trajectories: list[NDArray[np.floating]],
    segments: list[list[tuple[int, int]]],
    initial: list[tuple[float, float, float, float]],
    settings: FitSettings,
    device: torch.device,
) -> tuple[list[FlightFit], NDArray[np.float64]]:
    """Fits and final squared-residual sums from one start per rally."""
    spans = [
        (rally, start, end)
        for rally, rally_spans in enumerate(segments)
        for start, end in rally_spans
        if end - start >= settings.minimum_segment_frames
    ]
    if not spans:
        raise ValueError("No segment is long enough to fit")
    length = max(end - start for _, start, end in spans)
    dtype = torch.float64
    observed = torch.zeros(len(spans), length, 3, dtype=dtype)
    valid = torch.zeros(len(spans), length, dtype=torch.bool)
    for index, (rally, start, end) in enumerate(spans):
        observed[index, : end - start] = torch.as_tensor(
            trajectories[rally][start:end], dtype=dtype
        )
        valid[index, : end - start] = True
    observed, valid = observed.to(device), valid.to(device)
    weight = valid[..., None].to(dtype)
    owner = torch.tensor([rally for rally, _, _ in spans], device=device)
    rallies = len(trajectories)

    frame_time = settings.dt * settings.substeps
    start_velocity = (observed[:, 1] - observed[:, 0]) / frame_time
    start_velocity[:, 2] += 0.5 * settings.gravity * frame_time
    segment_params = torch.cat(
        (
            observed[:, 0],
            start_velocity,
            torch.zeros(len(spans), 3, dtype=dtype, device=device),
        ),
        dim=-1,
    )
    field_params = torch.tensor(
        [[math.log(d), math.log(m), wx, wy] for d, m, wx, wy in initial],
        dtype=dtype,
        device=device,
    )

    def simulate_one(params: torch.Tensor) -> torch.Tensor:
        """Positions ``(L,3)`` of one segment from its 13 parameters."""
        field = BallField(
            gravity=settings.gravity,
            k_drag=params[9].exp(),
            k_magnus=params[10].exp(),
            wind=torch.cat((params[11:13], params[11:12] * 0)),
        )
        positions, _ = integrate_flight(
            params[0:3],
            params[3:6],
            params[6:9] * _SPIN_UNIT,
            field,
            dt=settings.dt,
            substeps=settings.substeps,
            frames=length,
        )
        return positions

    simulate = vmap(simulate_one)
    jacobian = vmap(jacfwd(simulate_one))

    def residual_of(segment_p: torch.Tensor, field_p: torch.Tensor) -> torch.Tensor:
        params = torch.cat((segment_p, field_p[owner]), dim=-1)
        return (simulate(params) - observed) * weight

    def rally_loss(residual: torch.Tensor) -> torch.Tensor:
        per_segment = residual.square().sum(dim=(1, 2))
        return torch.zeros(rallies, dtype=dtype, device=device).index_add(
            0, owner, per_segment
        )

    damping = torch.full((rallies,), 1e-3, dtype=dtype, device=device)
    with torch.no_grad():
        residual = residual_of(segment_params, field_params)
        loss = rally_loss(residual)
        for iteration in range(settings.segment_iterations + settings.iterations):
            params = torch.cat((segment_params, field_params[owner]), dim=-1)
            jac = jacobian(params) * weight[..., None]  # (N, L, 3, 13)
            flat = jac.reshape(len(spans), -1, _SEGMENT + _FIELD)
            res = residual.reshape(len(spans), -1)
            normal = flat.transpose(1, 2) @ flat  # (N, 13, 13)
            gradient = (flat.transpose(1, 2) @ res[..., None])[..., 0]  # (N, 13)
            step_segment, step_field = _solve_rallies(
                normal,
                gradient,
                owner,
                rallies,
                damping,
                field_fixed=iteration < settings.segment_iterations,
            )
            candidate_segment = segment_params - step_segment
            candidate_field = field_params - step_field
            candidate_residual = residual_of(candidate_segment, candidate_field)
            candidate_loss = rally_loss(candidate_residual)
            better = candidate_loss < loss
            accept = better[owner]
            segment_params = torch.where(
                accept[:, None], candidate_segment, segment_params
            )
            field_params = torch.where(better[:, None], candidate_field, field_params)
            residual = torch.where(accept[:, None, None], candidate_residual, residual)
            loss = torch.where(better, candidate_loss, loss)
            damping = torch.where(better, damping / 3, damping * 4).clamp(1e-12, 1e8)

    residual_np = residual.cpu().numpy()
    results = []
    for rally, trajectory in enumerate(trajectories):
        per_frame = np.full((len(trajectory), 3), np.nan)
        for index, (owner_rally, start, end) in enumerate(spans):
            if owner_rally == rally:
                per_frame[start:end] = residual_np[index, : end - start]
        field = field_params[rally].cpu().numpy()
        results.append(
            FlightFit(
                residual=per_frame,
                k_drag=float(np.exp(field[0])),
                k_magnus=float(np.exp(field[1])),
                wind_xy=field[2:4].copy(),
            )
        )
    return results, loss.cpu().numpy()


def _solve_rallies(
    normal: torch.Tensor,
    gradient: torch.Tensor,
    owner: torch.Tensor,
    rallies: int,
    damping: torch.Tensor,
    *,
    field_fixed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped Gauss-Newton steps from per-segment normal-equation blocks.

    Each rally's system couples its segments only through the field, so the
    segment blocks are eliminated with a Schur complement.  With
    ``field_fixed`` the segments are solved independently and the field stays.
    """
    s, f = _SEGMENT, _FIELD
    seg_seg = normal[:, :s, :s]
    seg_field = normal[:, :s, s:]
    field_field = normal[:, s:, s:]
    g_seg, g_field = gradient[:, :s], gradient[:, s:]
    lam = damping[owner]
    eye_s = torch.eye(s, dtype=normal.dtype, device=normal.device)
    seg_damped = seg_seg + lam[:, None, None] * (
        eye_s * seg_seg.diagonal(dim1=1, dim2=2)[:, None, :] + 1e-12 * eye_s
    )
    inverse = torch.linalg.inv(seg_damped)
    if field_fixed:
        return (inverse @ g_seg[..., None])[..., 0], torch.zeros(
            rallies, f, dtype=normal.dtype, device=normal.device
        )
    # Schur complement on the field, summed over each rally's segments.
    reduced_n = field_field - seg_field.transpose(1, 2) @ inverse @ seg_field
    reduced_g = (
        g_field - (seg_field.transpose(1, 2) @ inverse @ g_seg[..., None])[..., 0]
    )
    system = torch.zeros(rallies, f, f, dtype=normal.dtype, device=normal.device)
    system = system.index_add(0, owner, reduced_n)
    rhs = torch.zeros(rallies, f, dtype=normal.dtype, device=normal.device)
    rhs = rhs.index_add(0, owner, reduced_g)
    eye_f = torch.eye(f, dtype=normal.dtype, device=normal.device)
    system = system + damping[:, None, None] * (
        eye_f * system.diagonal(dim1=1, dim2=2)[:, None, :] + 1e-12 * eye_f
    )
    step_field = torch.linalg.solve(system, rhs[..., None])[..., 0]
    step_segment = (
        inverse
        @ (g_seg - (seg_field @ step_field[owner][..., None])[..., 0])[..., None]
    )[..., 0]
    return step_segment, step_field
