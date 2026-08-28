"""BLCS module for 3D ball localization."""

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
from src.tasks.blcs.model_io import blcs_trajectory_prediction_to_physical
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

    from src.tasks.blcs.inference.predictor import BLCSPredictor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BLCSConfig:
    """Configuration for BLCS module.

    Attributes:
        checkpoint: Path to BLCS model checkpoint.
        device: Inference device.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).
        window_size: Maximum frames per model call. Long clips are split into
            overlapping windows so inference stays inside the trained
            ``seq_len_range`` instead of extrapolating RoPE far beyond it.
        window_overlap: Frames shared by consecutive windows (blended with
            center-peaked weights).

    """

    checkpoint: Path
    source: Literal["execute", "load"]
    device: str
    save_result: bool
    output_path: Path
    load_path: Path | None
    window_size: int
    window_overlap: int
    resolver: PathResolver
    court_keypoint_contract: CourtKeypointContract = field(
        default_factory=lambda: resolve_court_keypoint_contract("physical_v1")
    )

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "BLCS source='load' requires load_path; execute forbids it"
            )


@dataclass
class BLCSResult:
    """Result of BLCS inference.

    Attributes:
        ball_3d: Ball 3D position in court coords (T, 3), meters.
        visibility: Ball visibility mask (T,).

    """

    ball_3d: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    court_reference_provenance: CourtReferenceFrameProvenance = field(
        default_factory=build_physical_court_provenance
    )

    def to_dict(
        self,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> dict[str, object]:
        """Convert result to JSON-serializable dict."""
        data: dict[str, object] = {"ball_3d": self.ball_3d.tolist()}
        data["visibility"] = self.visibility.tolist()
        if court_keypoint_contract is not None:
            data = attach_court_keypoint_provenance(
                data,
                court_keypoint_contract,
                self.court_reference_provenance,
                location="BLCS result",
            )
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> BLCSResult:
        """Create result from dict."""
        provenance = (
            build_physical_court_provenance()
            if court_keypoint_contract is None
            else validate_court_keypoint_provenance(
                data,
                court_keypoint_contract,
                location="BLCS result",
            )
        )
        missing = {"ball_3d", "visibility"} - set(data)
        if missing:
            raise ValueError(
                f"BLCS result is missing required fields: {sorted(missing)}"
            )
        return cls(
            ball_3d=np.array(data["ball_3d"], dtype=np.float32),
            visibility=np.array(data["visibility"], dtype=np.bool_),
            court_reference_provenance=provenance,
        )

    def save(
        self,
        path: str | Path,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> None:
        """Save result to JSON file."""
        save_json(self.to_dict(court_keypoint_contract), path)
        LOGGER.info(f"Saved BLCS result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.
        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.ball_3d.ndim != 2 or self.ball_3d.shape[1] != 3:
            errors.append(f"ball_3d shape must be (T, 3), got {self.ball_3d.shape}")
        if not np.isfinite(self.ball_3d).all():
            errors.append("ball_3d contains non-finite values")
        if self.visibility.ndim != 1:
            errors.append(f"visibility shape must be (T,), got {self.visibility.shape}")
        if self.visibility.shape[0] != self.ball_3d.shape[0]:
            errors.append("visibility length does not match ball_3d length")
        if not np.isin(self.visibility, [0, 1, False, True]).all():
            errors.append("visibility must contain only 0 or 1")
        return len(errors) == 0, errors

    @classmethod
    def load(
        cls,
        path: str | Path,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> BLCSResult:
        """Load result from JSON file."""
        data = load_json(path)
        if not isinstance(data, dict):
            raise TypeError(f"BLCS result must be a JSON object: {path}")
        return cls.from_dict(data, court_keypoint_contract)


class BLCSModule(BasePipelineModule):
    """BLCS module for 3D ball localization.

    Predicts ball 3D trajectory from 2D ball positions
    and court keypoints.

    """

    def __init__(
        self,
        config: BLCSConfig,
    ) -> None:
        """Initialize the module.

        Args:
            config: BLCS configuration.

        """
        self.config = config
        self.checkpoint = self.config.checkpoint
        self.device = self.config.device
        self._predictor: BLCSPredictor | None = None

    def load(self) -> None:
        """Load the BLCS predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading BLCS model from {self.checkpoint}")

        from src.tasks.blcs.inference.predictor import BLCSPredictor

        self._predictor = BLCSPredictor.load_from_checkpoint(
            self.checkpoint,
            resolver=self.config.resolver,
            device=self.device,
            court_keypoints=self.config.court_keypoint_contract,
        )
        if self._predictor.input_profile != "multiview":
            raise ValueError(
                "tennis_scene BLCS requires model.io.input_profile='multiview', "
                f"got {self._predictor.input_profile!r}."
            )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._predictor is not None

    def process(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_],
        court_vis: NDArray[np.float32],
        *,
        court_keypoint_document: Mapping[str, object] | None = None,
        court_reference_provenance: CourtReferenceFrameProvenance | None = None,
    ) -> BLCSResult:
        """Run BLCS inference.

        Args:
            ball_uv: Ball 2D positions (N, T, 2), normalized [0, 1].
            court_kp: Court keypoints, shape (N, T, K, 2), normalized [0, 1].
            ball_vis: Ball visibility mask (N, T).
            court_vis: Court keypoint visibility, shape (N, T, K).

        Returns:
            BLCSResult with 3D ball trajectory.

        """
        # Check if we should load from pre-computed result
        if self.config.source == "load":
            load_path = self.config.load_path
            if load_path is None:
                raise RuntimeError("Validated load source is missing load_path")
            if load_path.is_file():
                LOGGER.info(
                    f"Loading BLCS result from {load_path} (skipping inference)"
                )
                return BLCSResult.load(
                    load_path,
                    self.config.court_keypoint_contract,
                )
            raise FileNotFoundError(f"BLCS artifact not found: {load_path}")

        input_provenance = self._validate_input_court_context(
            court_keypoint_document,
            court_reference_provenance,
        )

        if not self.is_loaded:
            self.load()
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("BLCS predictor is not loaded")

        if ball_uv.ndim != 3:
            raise ValueError(
                f"ball_uv must have shape (N, T, 2), got {ball_uv.shape}"
            )
        if ball_vis.ndim != 2:
            raise ValueError(
                f"ball_vis must have shape (N, T), got {ball_vis.shape}"
            )
        if court_kp.ndim != 4:
            raise ValueError(
                f"court_kp must have shape (N, T, K, 2), got {court_kp.shape}"
            )
        if court_vis.ndim != 3:
            raise ValueError(
                f"court_vis must have shape (N, T, K), got {court_vis.shape}"
            )
        LOGGER.info("Running BLCS ball localization...")

        num_frames = ball_uv.shape[1]

        slices = window_slices(
            num_frames, self.config.window_size, self.config.window_overlap
        )
        LOGGER.info(
            f"Running BLCS in {len(slices)} window(s) of <= "
            f"{self.config.window_size} frames (overlap {self.config.window_overlap})"
        )
        position_chunks = []
        for start, end in slices:
            if self.config.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR:
                prediction = predictor.predict_multiview_arrays(
                    ball_uv=ball_uv[:, start:end],
                    court_kp=court_kp[:, start:end],
                    ball_vis=ball_vis[:, start:end],
                    court_vis=court_vis[:, start:end],
                    denormalize=True,
                )
            else:
                prediction = predictor.predict_multiview_arrays(
                    ball_uv=ball_uv[:, start:end],
                    court_kp=court_kp[:, start:end],
                    ball_vis=ball_vis[:, start:end],
                    court_vis=court_vis[:, start:end],
                    denormalize=True,
                    court_keypoint_document=court_keypoint_document,
                    court_reference_provenance=(input_provenance,),
                )
                if prediction.court_reference_provenance != (input_provenance,):
                    raise CourtKeypointContractMismatchError(
                        "BLCS prediction provenance does not match its validated input."
                    )
                prediction = blcs_trajectory_prediction_to_physical(prediction)
            win_pos = prediction.position.squeeze(0).cpu().numpy()
            position_chunks.append((start, win_pos))

        ball_3d = blend_windows(position_chunks, num_frames).astype(np.float32)
        if ball_3d.shape != (num_frames, 3):
            raise ValueError(
                f"BLCS predictor position must have shape (T, 3), got {ball_3d.shape}"
            )
        output_visibility = np.asarray(ball_vis.any(axis=0), dtype=np.bool_)

        result = BLCSResult(
            ball_3d=ball_3d,
            visibility=output_visibility,
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
            location="tennis_scene BLCS input",
        )
        if provenance is None:
            if self.config.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
                raise MissingCourtKeypointMetadataError(
                    "tennis_scene BLCS camera_view_v2 input requires exact "
                    "Court reference provenance."
                )
            return build_physical_court_provenance()
        if provenance.contract != self.config.court_keypoint_contract:
            raise CourtKeypointContractMismatchError(
                "tennis_scene BLCS input provenance does not match runtime CourtKP20 "
                "contract."
            )
        return provenance
