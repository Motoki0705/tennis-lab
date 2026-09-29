"""Analytic tensors solely for allocation/gradient tests, never training data."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.losses import TrainingBatch
from src.tasks.ball_refiner.refiner_3d.diffusion.model import MixtureCondition


def analytic_memory_batch(fixture: Path, *, batch_size: int, frames: int, seed: int) -> TrainingBatch:
    if batch_size < 1 or frames < 16:
        raise ValueError("Analytic memory fixture needs B>=1 and T>=16")
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(frames, dtype=torch.float32) * (1001 / 60000)
    gravity = 9.81
    period = 6 / gravity
    phase = time.remainder(period)
    positions = torch.stack((-2 + 1.5 * time, 3 - 2 * time, 0.1 + 3 * phase - 0.5 * gravity * phase.square()), -1)
    positions = positions[None].expand(batch_size, -1, -1).clone()
    offsets = torch.randn((batch_size, 1, 64, 3), generator=generator) * 0.3
    means = positions[:, :, None] + offsets
    covariance = torch.diag(torch.tensor([1., 4., 0.25])).expand(batch_size, frames, 64, 3, 3).clone()
    topology = torch.tensor(list(product(range(4), repeat=3)))  # 0=absent, 1..3=mode
    subsets = topology > 0
    mass = torch.where(subsets, torch.tensor(0.98 / 3), torch.tensor(0.02)).prod(-1)
    weights = mass.expand(batch_size, frames, -1).clone()
    gap = slice((frames - min(64, frames // 2)) // 2, (frames + min(64, frames // 2)) // 2)
    covariance[:, gap] *= 16
    condition = MixtureCondition(
        means, covariance, weights, subsets.expand(batch_size, frames, -1, -1).clone(),
        torch.full((batch_size, frames), 0.02 ** 3), time[None].expand(batch_size, -1).clone(),
        torch.zeros((batch_size, frames), dtype=torch.bool),
    )
    camera_json = json.loads(fixture.read_text())["cameras"]
    if len(camera_json) != 3:
        raise ValueError("The memory fixture needs three cameras")
    matrices = torch.stack([torch.tensor(c["K"]) @ torch.cat((torch.tensor(c["R"]), torch.tensor(c["t"])[:, None]), -1) for c in camera_json])
    matrices = matrices[None].expand(batch_size, -1, -1, -1).clone()
    projected = torch.einsum("bvij,btj->bvti", matrices[..., :3], positions) + matrices[..., 3][:, :, None]
    if bool((projected[..., 2] <= 0).any()):
        raise ValueError("Analytic trajectory must be in front of fixture cameras")
    pixel = projected[..., :2] / projected[..., 2, None]
    alternatives = torch.tensor([[-5., 2.], [0., 0.], [5., -2.]])
    means_2d = pixel[..., None, :] + alternatives
    cov_2d = torch.tensor([[64., 12.], [12., 100.]]).expand(batch_size, 3, frames, 3, 2, 2).clone()
    cov_2d[:, :, gap] *= 16
    labels = torch.zeros((batch_size, frames, 2), dtype=torch.bool)
    labels[:, 0, 0] = True
    free = torch.ones((batch_size, frames), dtype=torch.bool)
    free[:, :6] = False
    for bounce in range(1, int(float(time[-1]) / period) + 1):
        frame = int(torch.abs(time - bounce * period).argmin())
        labels[:, frame, 1] = True
        free[:, max(0, frame - 5):frame + 6] = False
    return TrainingBatch(
        condition, positions, labels, free, matrices, means_2d, cov_2d,
        torch.tensor([0.2, 0.6, 0.2]).expand(batch_size, 3, frames, 3).clone(),
        torch.full((batch_size, 3, frames), 0.98), torch.full((batch_size,), gravity),
    )
