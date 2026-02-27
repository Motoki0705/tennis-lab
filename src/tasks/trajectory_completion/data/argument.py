"""Data augmentation utilities for UV trajectory completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from src.utils.data.augmentation import add_gaussian_noise


@dataclass(frozen=True)
class ArgumentConfig:
    """Configuration for trajectory data augmentation."""

    point_dropout_prob: float = 0.05
    event_dropout_prob: float = 0.0
    event_window: int = 2
    event_ratio: tuple[int, int] = (2, 1)  # naive:event (block count ratio)
    event_center_std: float | None = None
    noise_std: float = 0.01
    clamp_unit: bool = True
    outlier_prob: float = 0.0


class TrajectoryArgumenter:
    """Applies trajectory data augmentations based on config."""

    def __init__(self, cfg: Mapping | None) -> None:
        cfg = cfg or {}
        ratio = cfg.get("event_ratio", (2, 1))
        if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
            ratio_tuple = (int(ratio[0]), int(ratio[1]))
        else:
            ratio_tuple = (2, 1)
        self.config = ArgumentConfig(
            point_dropout_prob=float(cfg.get("point_dropout_prob", 0.05)),
            event_dropout_prob=float(cfg.get("event_dropout_prob", 0.0)),
            event_window=int(cfg.get("event_window", 2)),
            event_ratio=ratio_tuple,
            event_center_std=(
                None if cfg.get("event_center_std", None) is None else float(cfg.get("event_center_std"))
            ),
            noise_std=float(cfg.get("noise_std", 0.01)),
            clamp_unit=bool(cfg.get("clamp_unit", True)),
            outlier_prob=float(cfg.get("outlier_prob", 0.0)),
        )

    def __call__(
        self,
        ball_uv_gt: Tensor,
        ball_vis: Tensor,
        *,
        event_frames: Mapping[str, Tensor] | None = None,
        ratio: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Create (ball_uv_in, ball_obs_mask) from ground-truth UV and visibility."""
        ball_uv_in = ball_uv_gt.clone()
        ball_obs_mask = ball_vis.clone()

        if self.config.point_dropout_prob > 0:
            ball_obs_mask = self.apply_point_dropout(ball_obs_mask, self.config.point_dropout_prob)

        if self.config.event_dropout_prob > 0:
            use_ratio = ratio or self.config.event_ratio
            ball_obs_mask = self.apply_event_dropout(
                ball_obs_mask,
                event_frames=event_frames,
                ratio=use_ratio,
                window=self.config.event_window,
                drop_prob=self.config.event_dropout_prob,
                event_center_std=self.config.event_center_std,
            )

        if self.config.noise_std > 0:
            ball_uv_in = self.apply_noise(
                ball_uv_in,
                ball_obs_mask,
                noise_std=self.config.noise_std,
                clamp_unit=self.config.clamp_unit,
            )

        if self.config.outlier_prob > 0:
            ball_uv_in = self.apply_outlier(ball_uv_in, ball_obs_mask, self.config.outlier_prob)

        miss = ball_obs_mask <= 0
        if miss.any():
            ball_uv_in[miss] = 0.0

        return ball_uv_in, ball_obs_mask

    @staticmethod
    def apply_point_dropout(ball_obs_mask: Tensor, drop_prob: float) -> Tensor:
        """Randomly drop observed frames with probability p."""
        if drop_prob <= 0:
            return ball_obs_mask
        drop = (torch.rand(ball_obs_mask.shape[0], device=ball_obs_mask.device) < float(drop_prob)).to(
            ball_obs_mask.dtype
        )
        return ball_obs_mask * (1.0 - drop)

    @staticmethod
    def apply_event_dropout(
        ball_obs_mask: Tensor,
        *,
        event_frames: Mapping[str, Tensor] | None,
        ratio: tuple[int, int],
        window: int,
        drop_prob: float,
        event_center_std: float | None = None,
    ) -> Tensor:
        """Preferentially drop contiguous blocks around events with a naive:event block ratio.

        Notes:
            - The dropout rate is computed against the number of currently visible frames.
            - Masking is applied as contiguous blocks of length ``2*window+1`` (clipped at ends).
            - ``ratio`` controls the *number of blocks* drawn from event vs non-event regions,
              not the number of frames masked.
            - Event blocks are centered by sampling around the event time with a Gaussian
              distribution (highest probability at the event frame), constrained to
              ``[t-window, t+window]`` (and clipped to valid frame indices).
            - Block overlaps are allowed and are not de-duplicated.
        """
        if drop_prob <= 0:
            return ball_obs_mask

        obs_idx = torch.where(ball_obs_mask > 0)[0]
        if obs_idx.numel() == 0:
            return ball_obs_mask

        num_visible = int(obs_idx.numel())
        num_total_frames = int(round(float(drop_prob) * num_visible))
        if num_total_frames <= 0:
            return ball_obs_mask

        naive_ratio, event_ratio = ratio
        if naive_ratio < 0 or event_ratio < 0:
            naive_ratio, event_ratio = (2, 1)
        denom = naive_ratio + event_ratio
        if denom <= 0:
            naive_ratio, event_ratio = (2, 1)
            denom = 3

        length = int(ball_obs_mask.shape[0])
        block_len = max(1, 2 * int(window) + 1)
        num_blocks = max(1, int((num_total_frames + block_len - 1) // block_len))

        num_event_blocks = int(round(num_blocks * float(event_ratio) / float(denom)))
        num_naive_blocks = num_blocks - num_event_blocks

        # Enumerate event-centered blocks (bounce/shot). Sampling is without replacement
        # from the list of event centers, but duplicate centers are allowed if provided.
        event_centers: list[int] = []
        if event_frames:
            for key in ("bounce", "shot"):
                frames = event_frames.get(key)
                if frames is None or frames.numel() == 0:
                    continue
                frames = frames.to(device=ball_obs_mask.device).to(torch.long)
                event_centers.extend(int(t) for t in frames.tolist())

        if num_event_blocks > len(event_centers):
            num_naive_blocks += num_event_blocks - len(event_centers)
            num_event_blocks = len(event_centers)

        if num_event_blocks > 0:
            centers = torch.tensor(event_centers, device=ball_obs_mask.device, dtype=torch.long)
            perm = centers[torch.randperm(centers.numel(), device=ball_obs_mask.device)]
            chosen = perm[:num_event_blocks].tolist()
            for t in chosen:
                center = TrajectoryArgumenter._sample_event_block_center(
                    event_t=int(t),
                    length=length,
                    window=int(window),
                    std=event_center_std,
                    device=ball_obs_mask.device,
                )
                start = max(0, int(center) - int(window))
                end = min(length, int(center) + int(window) + 1)
                ball_obs_mask[start:end] = 0.0

        if num_naive_blocks > 0:
            if length <= 0:
                return ball_obs_mask
            event_candidates = TrajectoryArgumenter._expand_event_candidates(
                event_frames=event_frames,
                length=length,
                window=window,
                device=ball_obs_mask.device,
            )
            naive_pool = torch.where(~event_candidates)[0]
            if naive_pool.numel() == 0:
                centers = torch.randint(0, length, (int(num_naive_blocks),), device=ball_obs_mask.device)
            else:
                pick = torch.randint(0, int(naive_pool.numel()), (int(num_naive_blocks),), device=ball_obs_mask.device)
                centers = naive_pool[pick]
            centers = centers.tolist()
            for t in centers:
                start = max(0, int(t) - int(window))
                end = min(length, int(t) + int(window) + 1)
                ball_obs_mask[start:end] = 0.0

        return ball_obs_mask

    @staticmethod
    def _sample_event_block_center(
        *,
        event_t: int,
        length: int,
        window: int,
        std: float | None,
        device: torch.device,
    ) -> int:
        """Sample a block center around an event frame using a truncated Gaussian.

        The sampled center is constrained to ``[event_t-window, event_t+window]`` and then
        clipped into ``[0, length-1]``.
        """
        if length <= 0:
            return 0

        w = max(0, int(window))
        low = max(0, int(event_t) - w)
        high = min(int(length) - 1, int(event_t) + w)
        if low >= high:
            return int(low)

        sigma = float(std) if std is not None else max(1.0, float(w) / 2.0)
        if sigma <= 0:
            return int(min(max(event_t, low), high))

        mean = float(event_t)
        # Rejection sampling for a truncated normal; fall back to clamping if rare failures.
        for _ in range(8):
            x = torch.normal(mean=mean, std=sigma, size=(1,), device=device).round().to(torch.long).item()
            if low <= int(x) <= high:
                return int(x)
        x = int(torch.normal(mean=mean, std=sigma, size=(1,), device=device).round().to(torch.long).item())
        return int(min(max(x, low), high))

    @staticmethod
    def apply_noise(ball_uv_in: Tensor, ball_obs_mask: Tensor, *, noise_std: float, clamp_unit: bool) -> Tensor:
        """Add Gaussian noise to observed points."""
        obs = ball_obs_mask > 0
        if obs.any():
            noisy = add_gaussian_noise(ball_uv_in[obs], float(noise_std))
            if clamp_unit:
                noisy = noisy.clamp(0.0, 1.0)
            ball_uv_in = ball_uv_in.clone()
            ball_uv_in[obs] = noisy
        return ball_uv_in

    @staticmethod
    def apply_outlier(ball_uv_in: Tensor, ball_obs_mask: Tensor, outlier_prob: float) -> Tensor:
        """Replace observed points with random UVs with probability p."""
        obs = ball_obs_mask > 0
        if obs.any():
            outlier = (torch.rand(ball_obs_mask.shape[0], device=ball_obs_mask.device) < float(outlier_prob)) & obs
            if outlier.any():
                ball_uv_in = ball_uv_in.clone()
                ball_uv_in[outlier] = torch.rand(int(outlier.sum().item()), 2, device=ball_uv_in.device)
        return ball_uv_in

    @staticmethod
    def _expand_event_candidates(
        *,
        event_frames: Mapping[str, Tensor] | None,
        length: int,
        window: int,
        device: torch.device,
    ) -> Tensor:
        """Create a boolean mask for frames around events."""
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        if not event_frames:
            return mask
        for key in ("bounce", "shot"):
            frames = event_frames.get(key)
            if frames is None or frames.numel() == 0:
                continue
            frames = frames.to(device=device).to(torch.long)
            for t in frames.tolist():
                start = max(0, int(t) - int(window))
                end = min(length, int(t) + int(window) + 1)
                mask[start:end] = True
        return mask
