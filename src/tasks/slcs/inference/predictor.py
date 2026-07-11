"""SLCS predictor: batch inference and sliding-window clip inference.

``predict`` runs one collated window batch. ``predict_clip`` covers a full
clip camera with overlapping windows and aggregates per-frame outputs:
positions and log-scales are averaged over covering windows, rotations are
averaged in ``(cos, sin)`` space and renormalized. All outputs are CPU
tensors; ``*_meters`` / ``*_radians`` keys carry denormalized units.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.slcs.data.contract import ClipManifest
from src.tasks.slcs.data.dataset import (
    SLCSDataConfig,
    build_window_sample,
    collate_slcs,
    load_clip_arrays,
)
from src.tasks.slcs.data.dino_tokens import load_dino_tokens
from src.tasks.slcs.data.types import SLCSSample
from src.tasks.slcs.data.windows import plan_windows
from src.tasks.slcs.training.lightning_module import SLCSLightningModule
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class SLCSPredictor(BasePredictor):
    """Inference wrapper around a trained :class:`SLCSFusionModel`."""

    def __init__(self, lightning_module: SLCSLightningModule, device: torch.device) -> None:
        self.lightning_module = lightning_module.to(device)
        self.lightning_module.eval()
        self.model = lightning_module.model
        self.device = device

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> SLCSPredictor:
        """Load a predictor from an SLCS Lightning checkpoint."""
        module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            SLCSLightningModule,
            device,
            strict=bool(kwargs.pop("strict", False)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        return cls(module, resolved_device)

    # ------------------------------------------------------------------

    @torch.no_grad()  # type: ignore[untyped-decorator, unused-ignore]
    def predict(
        self,
        batch: dict[str, Tensor],
        *,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Run one collated SLCS batch; returns CPU tensors.

        Keys: ``player_position``, ``player_rotation``, ``ball_position``,
        ``*_log_b`` (normalized domain), plus ``player_position_meters``,
        ``player_yaw_radians``, ``ball_position_meters`` and per-frame
        ``*_sigma`` uncertainty when ``denormalize=True``.
        """
        moved = {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        outputs = self.lightning_module.forward_batch(moved)
        result = {key: value.detach().cpu() for key, value in outputs.items()}
        if denormalize:
            result.update(self._denormalize(result))
        return result

    def _denormalize(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        scale = torch.tensor(list(COURT_COORD_SCALE_XYZ), dtype=torch.float32)
        scale_mean = float(scale.mean().item())
        rotation = torch.nn.functional.normalize(outputs["player_rotation"], dim=-1)
        return {
            "player_position_meters": outputs["player_position"] * scale,
            "player_yaw_radians": torch.atan2(rotation[..., 1], rotation[..., 0]),
            "ball_position_meters": outputs["ball_position"] * scale,
            "player_position_sigma_m": outputs["player_position_log_b"].exp()
            * scale_mean,
            "player_rotation_sigma_rad": outputs["player_rotation_log_b"].exp(),
            "ball_position_sigma_m": outputs["ball_position_log_b"].exp() * scale_mean,
        }

    # ------------------------------------------------------------------

    @torch.no_grad()  # type: ignore[untyped-decorator, unused-ignore]
    def predict_clip(
        self,
        clip_dir: str | Path,
        camera_id: str,
        *,
        data_config: SLCSDataConfig,
        stride: int | None = None,
        batch_size: int = 4,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict the full timeline of one clip camera.

        Windows follow ``data_config.window_size`` with ``stride`` (default:
        ``data_config.eval_stride``); overlapping predictions are averaged.

        Returns full-length tensors: ``player_position (P, T, 3)``,
        ``player_rotation (P, T, 2)``, ``ball_position (T, 3)``, the
        corresponding ``*_log_b`` averages, ``coverage (T,)`` (windows per
        frame) and denormalized keys as in :meth:`predict`.
        """
        manifest = ClipManifest.load(clip_dir)
        clip = load_clip_arrays(manifest, config=data_config)
        spec = data_config.dino_spec
        assert spec is not None  # enforced by SLCSDataConfig
        dino_arrays = (
            load_dino_tokens(manifest, camera_id, expected_spec=spec)[:2]
            if data_config.require_dino
            else None
        )
        camera_index = manifest.camera_index(camera_id)
        plans = plan_windows(
            clip.num_frames,
            window_size=data_config.window_size,
            stride=int(stride if stride is not None else data_config.eval_stride),
        )
        samples: list[SLCSSample] = [
            build_window_sample(
                clip,
                camera_index=camera_index,
                plan=plan,
                dino_arrays=dino_arrays,
                empty_dino_shape=(spec.num_tokens, spec.embed_dim),
            )
            for plan in plans
        ]

        num_frames = clip.num_frames
        num_players = data_config.num_players
        acc: dict[str, Tensor] = {
            "player_position": torch.zeros(num_players, num_frames, 3),
            "player_rotation": torch.zeros(num_players, num_frames, 2),
            "ball_position": torch.zeros(num_frames, 3),
            "player_position_log_b": torch.zeros(num_players, num_frames),
            "player_rotation_log_b": torch.zeros(num_players, num_frames),
            "ball_position_log_b": torch.zeros(num_frames),
        }
        coverage = torch.zeros(num_frames)

        for start in range(0, len(samples), max(1, batch_size)):
            chunk = samples[start : start + max(1, batch_size)]
            batch = collate_slcs(chunk)
            outputs = self.predict(batch, denormalize=False)
            for i, plan in enumerate(plans[start : start + len(chunk)]):
                t0, length = plan.start, plan.length
                sl = slice(t0, t0 + length)
                acc["player_position"][:, sl] += outputs["player_position"][i, :, :length]
                acc["player_rotation"][:, sl] += outputs["player_rotation"][i, :, :length]
                acc["ball_position"][sl] += outputs["ball_position"][i, :length]
                acc["player_position_log_b"][:, sl] += outputs["player_position_log_b"][
                    i, :, :length
                ]
                acc["player_rotation_log_b"][:, sl] += outputs["player_rotation_log_b"][
                    i, :, :length
                ]
                acc["ball_position_log_b"][sl] += outputs["ball_position_log_b"][i, :length]
                coverage[sl] += 1.0

        if bool((coverage == 0).any()):
            uncovered = torch.nonzero(coverage == 0).flatten().tolist()
            raise RuntimeError(
                f"{manifest.clip_id}: frames without window coverage: {uncovered[:10]} "
                f"(total {len(uncovered)}). This indicates broken window planning."
            )
        denom_frames = coverage.clamp(min=1.0)
        result: dict[str, Tensor] = {
            "player_position": acc["player_position"] / denom_frames[None, :, None],
            "player_rotation": torch.nn.functional.normalize(
                acc["player_rotation"] / denom_frames[None, :, None], dim=-1
            ),
            "ball_position": acc["ball_position"] / denom_frames[:, None],
            "player_position_log_b": acc["player_position_log_b"] / denom_frames[None, :],
            "player_rotation_log_b": acc["player_rotation_log_b"] / denom_frames[None, :],
            "ball_position_log_b": acc["ball_position_log_b"] / denom_frames,
            "coverage": coverage,
        }
        if denormalize:
            result.update(self._denormalize(result))
        return result


__all__ = ["SLCSPredictor"]
