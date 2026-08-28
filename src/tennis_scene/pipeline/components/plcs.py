"""PLCS module for 3D player localization."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    build_physical_court_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import validate_model_artifact_court_keypoint_contract
from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.tennis_scene.schema import (
    attach_court_keypoint_provenance,
    validate_court_keypoint_provenance,
)
from src.utils.configuration import PathResolver
from src.utils.inference.windowed import blend_windows, window_slices
from src.utils.io import load_json, save_json

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tasks.plcs.inference.predictor import PLCSPredictor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PLCSConfig:
    """Configuration for PLCS module.

    Attributes:
        checkpoint: Path to PLCS model checkpoint.
        device: Inference device.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).
        window_size: Maximum frames per model call. Long clips are split into
            overlapping windows so inference stays inside the trained
            ``seq_len_range`` instead of extrapolating RoPE far beyond it.
        window_overlap: Frames shared by consecutive windows (blended with
            center-peaked weights).
        human_vis_threshold: Detector confidences below this become invisible
            (0). Real pose detectors emit continuous confidences and keep
            hallucinated coordinates for occluded joints, while training
            visibility is binary — thresholding restores that contract.
    """

    checkpoint: Path
    source: Literal["execute", "load"]
    device: str
    save_result: bool
    output_path: Path
    load_path: Path | None
    window_size: int
    window_overlap: int
    human_vis_threshold: float
    resolver: PathResolver
    court_keypoint_contract: CourtKeypointContract = field(
        default_factory=lambda: resolve_court_keypoint_contract("physical_v1")
    )

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "PLCS source='load' requires load_path; execute forbids it"
            )


@dataclass
class PLCSResult:
    """Result of PLCS inference.

    Attributes:
        position: Player 3D positions in court coords, shape (P, T, 3), meters.
        yaw: Player yaw angles, shape (P, T), radians.
        track_ids: Track IDs aligned to player axis, shape (P,).
    """

    position: NDArray[np.float32]
    yaw: NDArray[np.float32]
    track_ids: NDArray[np.int32]
    court_reference_provenance: CourtReferenceFrameProvenance = field(
        default_factory=build_physical_court_provenance
    )

    def to_dict(
        self,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "position": self.position.tolist(),
            "yaw": self.yaw.tolist(),
        }
        result["track_ids"] = self.track_ids.tolist()
        if court_keypoint_contract is not None:
            result = attach_court_keypoint_provenance(
                result,
                court_keypoint_contract,
                self.court_reference_provenance,
                location="PLCS result",
            )
        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> PLCSResult:
        provenance = (
            build_physical_court_provenance()
            if court_keypoint_contract is None
            else validate_court_keypoint_provenance(
                data,
                court_keypoint_contract,
                location="PLCS result",
            )
        )
        missing = {"position", "yaw", "track_ids"} - set(data)
        if missing:
            raise ValueError(
                f"PLCS result is missing required fields: {sorted(missing)}"
            )
        position = np.array(data["position"], dtype=np.float32)
        yaw = np.array(data["yaw"], dtype=np.float32)

        track_ids = np.array(data["track_ids"], dtype=np.int32)

        return cls(
            position=position,
            yaw=yaw,
            track_ids=track_ids,
            court_reference_provenance=provenance,
        )

    def save(
        self,
        path: str | Path,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> None:
        save_json(self.to_dict(court_keypoint_contract), path)
        LOGGER.info(f"Saved PLCS result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []

        if self.position.ndim != 3 or self.position.shape[-1] != 3:
            errors.append(
                f"position shape must be (P, T, 3), got {self.position.shape}"
            )
        if self.yaw.ndim != 2:
            errors.append(f"yaw shape must be (P, T), got {self.yaw.shape}")

        if (
            self.position.ndim == 3
            and self.yaw.ndim == 2
            and self.position.shape[:2] != self.yaw.shape
        ):
            errors.append("position and yaw shapes are inconsistent on (P, T)")

        if self.track_ids.ndim != 1:
            errors.append(f"track_ids shape must be (P,), got {self.track_ids.shape}")
        elif self.track_ids.shape[0] != self.position.shape[0]:
            errors.append("track_ids length does not match player count")

        if not np.isfinite(self.position).all():
            errors.append("position contains non-finite values")
        if not np.isfinite(self.yaw).all():
            errors.append("yaw contains non-finite values")

        return len(errors) == 0, errors

    @classmethod
    def load(
        cls,
        path: str | Path,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> PLCSResult:
        data = load_json(path)
        if not isinstance(data, dict):
            raise TypeError(f"PLCS result must be a JSON object: {path}")
        return cls.from_dict(data, court_keypoint_contract)


class PLCSModule(BasePipelineModule):
    """PLCS module for 3D player localization."""

    def __init__(self, config: PLCSConfig) -> None:
        self.config = config
        self.checkpoint = self.config.checkpoint
        self.device = self.config.device
        self._predictor: PLCSPredictor | None = None

    def load(self) -> None:
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading PLCS model from {self.checkpoint}")
        from src.tasks.plcs.inference.predictor import PLCSPredictor

        self._predictor = PLCSPredictor.load_from_checkpoint(
            self.checkpoint,
            resolver=self.config.resolver,
            device=self.device,
            court_keypoint_contract=self.config.court_keypoint_contract,
        )
        self._predictor.require_input_profile("multiview")

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def process(
        self,
        human_kp_2d: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        human_kp_vis: NDArray[np.float32],
        court_vis: NDArray[np.float32],
        track_ids: NDArray[np.int32],
        *,
        court_keypoint_document: Mapping[str, object] | None = None,
        court_reference_provenance: CourtReferenceFrameProvenance | None = None,
    ) -> PLCSResult:
        """Run PLCS inference.

        Args:
            human_kp_2d: Human 2D keypoints, shape (P, N, T, 17, 2), normalized [0, 1].
            court_kp: Court keypoints, shape (N, T, K, 2), normalized [0, 1].
            human_kp_vis: Human keypoint visibility, shape (P, N, T, 17).
            court_vis: Court keypoint visibility, shape (N, T, K).
            track_ids: Track IDs aligned with P.

        Returns:
            PLCSResult with position/yaw in (P, T, ...).
        """
        if self.config.source == "load":
            load_path = self.config.load_path
            if load_path is None:
                raise RuntimeError("Validated load source is missing load_path")
            if load_path.is_file():
                LOGGER.info(
                    f"Loading PLCS result from {load_path} (skipping inference)"
                )
                return PLCSResult.load(
                    load_path,
                    self.config.court_keypoint_contract,
                )
            raise FileNotFoundError(f"PLCS artifact not found: {load_path}")

        input_provenance = self._validate_input_court_context(
            court_keypoint_document,
            court_reference_provenance,
        )
        direct_document = (
            None if court_keypoint_document is None else dict(court_keypoint_document)
        )

        if not self.is_loaded:
            self.load()
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("PLCS predictor is not loaded")

        if human_kp_2d.ndim != 5:
            raise ValueError(
                "human_kp_2d must have shape (P, N, T, 17, 2), "
                f"got {human_kp_2d.shape}"
            )
        if human_kp_vis.ndim != 4:
            raise ValueError(
                "human_kp_vis must have shape (P, N, T, 17), "
                f"got {human_kp_vis.shape}"
            )
        if court_kp.ndim != 4:
            raise ValueError(
                f"court_kp must have shape (N, T, K, 2), got {court_kp.shape}"
            )
        if court_vis.ndim != 3:
            raise ValueError(
                f"court_vis must have shape (N, T, K), got {court_vis.shape}"
            )
        num_players, num_cameras, num_frames = human_kp_2d.shape[:3]
        if track_ids.shape != (num_players,):
            raise ValueError(
                f"track_ids must have shape ({num_players},), got {track_ids.shape}"
            )

        LOGGER.info(
            "Running PLCS player localization for "
            f"{num_players} players and {num_cameras} cameras..."
        )

        binary_vis = human_kp_vis >= self.config.human_vis_threshold
        dropped = float((human_kp_vis > 0).mean() - binary_vis.mean())
        LOGGER.info(
            "Thresholded human keypoint confidence at "
            f"{self.config.human_vis_threshold} (dropped {dropped:.1%} of joints)"
        )
        padding_mask = np.zeros(
            (num_players, num_cameras, num_frames), dtype=np.bool_
        )

        slices = window_slices(
            num_frames, self.config.window_size, self.config.window_overlap
        )
        LOGGER.info(
            f"Running PLCS in {len(slices)} window(s) of <= "
            f"{self.config.window_size} frames (overlap {self.config.window_overlap})"
        )
        position_chunks: list[tuple[int, NDArray[np.float64]]] = []
        yaw_vec_chunks: list[tuple[int, NDArray[np.float64]]] = []
        for start, end in slices:
            if self.config.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR:
                prediction = predictor.predict_multiview_observations(
                    human_kp=human_kp_2d[:, :, start:end],
                    court_kp=court_kp[:, start:end],
                    human_vis=binary_vis[:, :, start:end],
                    padding_mask=padding_mask[:, :, start:end],
                    court_vis=court_vis[:, start:end],
                )
            else:
                prediction = predictor.predict_multiview_observations(
                    human_kp=human_kp_2d[:, :, start:end],
                    court_kp=court_kp[:, start:end],
                    human_vis=binary_vis[:, :, start:end],
                    padding_mask=padding_mask[:, :, start:end],
                    court_vis=court_vis[:, start:end],
                    court_keypoint_metadata=direct_document,
                    court_reference_provenance=input_provenance,
                )
                if prediction.court_reference_provenance != (input_provenance,):
                    raise CourtKeypointContractMismatchError(
                        "PLCS prediction provenance does not match its validated input."
                    )
            win_pos = prediction.position_meters
            win_yaw = prediction.yaw_radians
            position_chunks.append((start, win_pos.transpose(1, 0, 2)))
            yaw_vec = np.stack([np.sin(win_yaw), np.cos(win_yaw)], axis=-1)
            yaw_vec_chunks.append((start, yaw_vec.transpose(1, 0, 2)))

        positions = (
            blend_windows(position_chunks, num_frames)
            .transpose(1, 0, 2)
            .astype(np.float32)
        )
        yaw_vec_blend = blend_windows(yaw_vec_chunks, num_frames).transpose(1, 0, 2)
        yaws = np.arctan2(yaw_vec_blend[..., 0], yaw_vec_blend[..., 1]).astype(
            np.float32
        )
        if positions.shape != (num_players, num_frames, 3):
            raise ValueError(
                "PLCS predictor position_meters must have shape (P, T, 3), "
                f"got {positions.shape}"
            )
        if yaws.shape != (num_players, num_frames):
            raise ValueError(
                f"PLCS predictor yaw_radians must have shape (P, T), got {yaws.shape}"
            )

        result = PLCSResult(
            position=positions,  # (P, T, 3)
            yaw=yaws,  # (P, T)
            track_ids=track_ids.astype(np.int32),
            court_reference_provenance=input_provenance,
        )

        if self.config.save_result:
            result.save(
                self.config.output_path,
                self.config.court_keypoint_contract,
            )

        return result

    def _validate_input_court_context(
        self,
        document: Mapping[str, object] | None,
        provenance: CourtReferenceFrameProvenance | None,
    ) -> CourtReferenceFrameProvenance:
        """Fail closed on direct v2 input before loading or invoking a model."""
        validate_model_artifact_court_keypoint_contract(
            {} if document is None else document,
            self.config.court_keypoint_contract,
            location="tennis_scene PLCS input",
        )
        if provenance is None:
            if self.config.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
                raise MissingCourtKeypointMetadataError(
                    "tennis_scene PLCS camera_view_v2 input requires exact "
                    "Court reference provenance."
                )
            return build_physical_court_provenance()
        if provenance.contract != self.config.court_keypoint_contract:
            raise CourtKeypointContractMismatchError(
                "tennis_scene PLCS input provenance does not match runtime CourtKP20 "
                "contract."
            )
        return provenance
