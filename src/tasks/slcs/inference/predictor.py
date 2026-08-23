"""SLCS predictor for batched and sliding-window clip inference.

Batch and overlapping-window clip inference return distinct typed CPU outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import ParamSpec, TypeVar

import torch
from torch import Tensor

from src.tasks.base.data import CourtCoordinateContractMismatchError
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO
from src.tasks.slcs.data.dataset import (
    SLCSDataConfig,
    build_window_sample,
    collate_slcs,
    load_clip_arrays,
)
from src.tasks.slcs.data.dino_tokens import load_dino_tokens
from src.tasks.slcs.data.types import SLCSSample
from src.tasks.slcs.data.windows import plan_windows
from src.tasks.slcs.model_io import (
    SLCSClipPrediction,
    SLCSDecodedOutput,
    SLCSModelIOAdapter,
    SLCSRawOutput,
    SLCSTrainingTargets,
    load_slcs_checkpoint_mapping,
    prepare_slcs_checkpoint_config,
)
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.tasks.slcs.training.lightning_module import SLCSLightningModule
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.configuration import PathResolver
from src.utils.device import resolve_device
from src.utils.schema.court_normalization import CourtCoordinateNormalization

_P = ParamSpec("_P")
_R = TypeVar("_R")
_no_grad: Callable[[Callable[_P, _R]], Callable[_P, _R]] = torch.no_grad()


class SLCSPredictor(BasePredictor[SLCSDecodedOutput]):
    """Inference wrapper around a trained :class:`SLCSFusionModel`."""

    def __init__(
        self, lightning_module: SLCSLightningModule, device: torch.device
    ) -> None:
        self.lightning_module = lightning_module.to(device)
        self.lightning_module.eval()
        self.model: SLCSFusionModel = lightning_module.model
        self.model_io: BoundModelIO[
            Mapping[str, object], SLCSRawOutput, SLCSDecodedOutput
        ] = lightning_module.model_io
        self.model_adapter: SLCSModelIOAdapter = lightning_module.model_adapter
        self.court_coordinate_normalization = (
            self.model_adapter.court_coordinate_normalization
        )
        self.device = device

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        court_coordinate_normalization: CourtCoordinateNormalization,
        device: str | torch.device,
        strict: bool,
        weights_only: bool,
    ) -> SLCSPredictor:
        """Load a predictor from an SLCS Lightning checkpoint."""
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        resolved_device = resolve_device(device)
        checkpoint = load_slcs_checkpoint_mapping(checkpoints[0])
        checkpoint_config = prepare_slcs_checkpoint_config(
            checkpoint,
            court_coordinate_normalization,
            location=str(checkpoints[0]),
        )
        module = SLCSLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
            config=checkpoint_config,
            strict=strict,
            weights_only=weights_only,
        )
        return cls(module, resolved_device)

    # ------------------------------------------------------------------

    @_no_grad
    def predict(
        self,
        batch: dict[str, Tensor],
    ) -> SLCSDecodedOutput:
        """Validate and run one collated batch; return typed CPU predictions."""
        moved = {key: value.to(self.device) for key, value in batch.items()}
        output: SLCSDecodedOutput = self.model_io.run(moved)
        return output.detached_cpu()

    @_no_grad
    def predict_with_targets(
        self, batch: dict[str, Tensor]
    ) -> tuple[SLCSDecodedOutput, SLCSTrainingTargets]:
        """Validate all targets before executing an evaluation model call."""
        moved = {key: value.to(self.device) for key, value in batch.items()}
        call = self.model_io.build_call(moved)
        targets = self.model_adapter.build_training_targets(moved)
        output = self.model_io.decode_output(self.model_io.execute_call(call))
        return output.detached_cpu(), targets.detached_cpu()

    # ------------------------------------------------------------------

    @_no_grad
    def predict_clip(
        self,
        clip_dir: str | Path,
        camera_id: str,
        *,
        data_config: SLCSDataConfig,
        stride: int,
        batch_size: int,
    ) -> SLCSClipPrediction:
        """Predict the full timeline of one clip camera.

        Windows follow ``data_config.window_size`` with the explicit ``stride``;
        overlapping predictions are averaged.

        The returned ``normalized`` and ``physical`` fields contain the
        aggregated predictions. ``coverage (T,)`` records the number of
        windows contributing to each frame.
        """
        manifest = ClipManifest.load(clip_dir)
        if (
            data_config.court_coordinate_normalization
            != self.court_coordinate_normalization
        ):
            raise CourtCoordinateContractMismatchError(
                "SLCS clip dataset normalization "
                f"{data_config.court_coordinate_normalization.version!r}/"
                f"{data_config.court_coordinate_normalization.scale_xyz!r} does "
                "not match predictor normalization "
                f"{self.court_coordinate_normalization.version!r}/"
                f"{self.court_coordinate_normalization.scale_xyz!r}."
            )
        clip = load_clip_arrays(manifest, config=data_config)
        spec = data_config.dino_spec
        dino_arrays = (
            load_dino_tokens(manifest, camera_id, expected_spec=spec)[:2]
            if data_config.require_dino
            else None
        )
        camera_index = manifest.camera_index(camera_id)
        plans = plan_windows(
            clip.num_frames,
            window_size=data_config.window_size,
            stride=stride,
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

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = collate_slcs(chunk)
            outputs = self.predict(batch)
            for i, plan in enumerate(plans[start : start + len(chunk)]):
                t0, length = plan.start, plan.length
                sl = slice(t0, t0 + length)
                acc["player_position"][:, sl] += outputs.player_position[
                    i, :, :length
                ]
                acc["player_rotation"][:, sl] += outputs.player_rotation[
                    i, :, :length
                ]
                acc["ball_position"][sl] += outputs.ball_position[i, :length]
                acc["player_position_log_b"][
                    :, sl
                ] += outputs.player_position_log_b[i, :, :length]
                acc["player_rotation_log_b"][
                    :, sl
                ] += outputs.player_rotation_log_b[i, :, :length]
                acc["ball_position_log_b"][sl] += outputs.ball_position_log_b[
                    i, :length
                ]
                coverage[sl] += 1.0

        if bool((coverage == 0).any()):
            uncovered = torch.nonzero(coverage == 0).flatten().tolist()
            raise RuntimeError(
                f"{manifest.clip_id}: frames without window coverage: {uncovered[:10]} "
                f"(total {len(uncovered)}). This indicates broken window planning."
            )
        denom_frames = coverage.clamp(min=1.0)
        normalized = SLCSDecodedOutput(
            player_position=acc["player_position"] / denom_frames[None, :, None],
            player_rotation=torch.nn.functional.normalize(
                acc["player_rotation"] / denom_frames[None, :, None], dim=-1
            ),
            ball_position=acc["ball_position"] / denom_frames[:, None],
            player_position_log_b=acc["player_position_log_b"]
            / denom_frames[None, :],
            player_rotation_log_b=acc["player_rotation_log_b"]
            / denom_frames[None, :],
            ball_position_log_b=acc["ball_position_log_b"] / denom_frames,
        )
        return SLCSClipPrediction(
            normalized=normalized,
            physical=self.model_adapter.to_physical(normalized),
            coverage=coverage,
        )


__all__ = ["SLCSPredictor"]
