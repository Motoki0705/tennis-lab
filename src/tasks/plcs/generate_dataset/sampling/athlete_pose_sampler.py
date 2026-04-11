"""AthletePose3D motion sampler for PLCS dataset generation.

Loads pre-computed 3D pose sequences from AthletePose3D (Qualisys MoCap)
and converts them to COCO17-format world-coordinate joint sequences suitable
for the PLCS scene-generation pipeline.

The AthletePose3D data uses Human3.6M 17-joint convention in normalised
image-like coordinates.  This module handles:
- Joint re-ordering from H3.6M → COCO17
- Coordinate-system conversion (screen Y-down → world Z-up)
- Metric scaling (normalised units → metres)
- Per-frame yaw estimation from shoulder/hip orientation
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.utils.schema.player import (
    FACE_KEYPOINT_OFFSETS,
    H36M_HEAD_JOINT,
    H36M_TO_COCO17_MAPPING,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# Scale factor: converts normalised AthletePose3D units to metres.
#
# AthletePose3D MoCap data is normalised so that the average pelvis→head-top
# distance is ≈0.156 units.  An average person's pelvis→head-top span is
# ≈0.83 m, giving  0.83 / 0.156 ≈ 5.3 m/unit.
# ---------------------------------------------------------------------------
_DEFAULT_SCALE_FACTOR: float = 5.3

# Default assumed FPS for AthletePose3D sequences (MoCap 200 fps down-sampled
# to match the typical PLCS working rate).
_DEFAULT_FPS: int = 30


@dataclass
class AthletePoseMotion:
    """A single 3D-pose sequence loaded from AthletePose3D.

    Attributes
    ----------
    joints_coco17 : np.ndarray
        (T, 17, 3) COCO17 joints in root-relative world coordinates (metres).
        Coordinate axes: X = sideline, Y = forward, Z = up.
    yaw : np.ndarray
        (T,) per-frame facing direction (radians, in world XY-plane).
    source_path : str
        Filesystem path of the originating pkl file.
    num_frames : int
        Number of frames in the sequence.
    fps : int
        Nominal frame rate.
    """

    joints_coco17: np.ndarray  # (T, 17, 3) root-relative, metres
    yaw: np.ndarray  # (T,)
    source_path: str = ""
    num_frames: int = 0
    fps: int = _DEFAULT_FPS


class AthletePose3DSampler:
    """Sample 3D pose sequences from the AthletePose3D dataset.

    Parameters
    ----------
    config : DictConfig | None
        Hydra configuration.  Recognised keys under ``athlete_pose``:
        ``data_dir``, ``split``, ``scale_factor``, ``fps``.
    data_dir : str | Path | None
        Root directory (containing ``frame_81/{train,test}/``).  Overrides
        the config value.
    split : str
        ``"train"`` or ``"test"``.
    scale_factor : float
        Multiplier from normalised units to metres.
    fps : int
        Nominal frame rate recorded in metadata.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        data_dir: str | Path | None = None,
        split: str = "train",
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        fps: int = _DEFAULT_FPS,
    ) -> None:
        cfg = config.get("athlete_pose", {}) if config is not None else {}

        self.data_dir = Path(data_dir or cfg.get("data_dir", "data/AthletePose3D/pose_3d_v3"))
        self.split = cfg.get("split", split)
        self.scale_factor = float(cfg.get("scale_factor", scale_factor))
        self.fps = int(cfg.get("fps", fps))

        split_dir = self.data_dir / f"frame_81/{self.split}"
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"AthletePose3D split directory not found: {split_dir}"
            )

        self.files: list[Path] = sorted(split_dir.glob("*.pkl"))
        if not self.files:
            raise FileNotFoundError(
                f"No .pkl files found in {split_dir}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.files)

    def sample(self) -> AthletePoseMotion:
        """Return a randomly sampled motion sequence."""
        path = random.choice(self.files)
        return self.load(path)

    def load(self, path: str | Path) -> AthletePoseMotion:
        """Load a specific pkl file and convert to world-coordinate COCO17."""
        path = Path(path)
        with path.open("rb") as fh:
            raw: dict[str, np.ndarray] = pickle.load(fh)  # noqa: S301

        h36m_joints: np.ndarray = raw["data_label"].astype(np.float32)  # (T, 17, 3)
        T = h36m_joints.shape[0]

        # 1. Make root-relative (pelvis = joint 0)
        pelvis = h36m_joints[:, 0:1, :].copy()  # (T, 1, 3)
        rel = h36m_joints - pelvis  # (T, 17, 3)

        # 2. Convert coordinate system: screen(X,Y,Z) → world(X,Y,Z):
        #      world_X =  screen_X  (horizontal)
        #      world_Y =  screen_Z  (depth → forward)
        #      world_Z = -screen_Y  (flip: screen-down → up)
        world_rel = np.empty_like(rel)
        world_rel[..., 0] = rel[..., 0]
        world_rel[..., 1] = rel[..., 2]
        world_rel[..., 2] = -rel[..., 1]

        # 3. Scale to metres
        world_rel *= self.scale_factor

        # 4. Map H3.6M joints → COCO17
        coco17 = self._h36m_to_coco17(world_rel, h36m_joints)

        # 5. Estimate per-frame yaw from shoulder/hip orientation
        yaw = self._estimate_yaw(coco17)

        return AthletePoseMotion(
            joints_coco17=coco17,
            yaw=yaw,
            source_path=str(path),
            num_frames=T,
            fps=self.fps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _h36m_to_coco17(
        world_rel: np.ndarray,
        h36m_joints_raw: np.ndarray,
    ) -> np.ndarray:
        """Convert H3.6M 17 joints to COCO17 format.

        Body joints are mapped directly.  Face keypoints (nose, eyes, ears)
        are synthesised from the head position using ``FACE_KEYPOINT_OFFSETS``.

        Parameters
        ----------
        world_rel : np.ndarray
            (T, 17, 3) root-relative joints already in world coords (metres).
        h36m_joints_raw : np.ndarray
            (T, 17, 3) original H3.6M joints (only used for head direction).

        Returns
        -------
        np.ndarray
            (T, 17, 3) COCO17 joints in metres, root-relative world coords.
        """
        T = world_rel.shape[0]
        coco17 = np.zeros((T, 17, 3), dtype=np.float32)

        # Map body joints
        for coco_idx, h36m_idx in H36M_TO_COCO17_MAPPING.items():
            if h36m_idx >= 0:
                coco17[:, coco_idx, :] = world_rel[:, h36m_idx, :]

        # Head position (world coords) for face-keypoint synthesis
        head_pos = world_rel[:, H36M_HEAD_JOINT, :]  # (T, 3)

        # Use identity yaw (=0) for face offsets in root-relative canonical
        # form; the actual yaw rotation is applied later by SceneGenerator.
        cos_yaw = np.ones(T, dtype=np.float32)
        sin_yaw = np.zeros(T, dtype=np.float32)

        for coco_idx, offset in FACE_KEYPOINT_OFFSETS.items():
            off = np.array(offset, dtype=np.float32)
            rotated = np.stack(
                [
                    off[0] * cos_yaw - off[1] * sin_yaw,
                    off[0] * sin_yaw + off[1] * cos_yaw,
                    np.full(T, off[2], dtype=np.float32),
                ],
                axis=1,
            )
            coco17[:, coco_idx, :] = head_pos + rotated

        return np.asarray(coco17)

    @staticmethod
    def _estimate_yaw(coco17: np.ndarray) -> np.ndarray:
        """Estimate per-frame facing yaw from shoulder and hip midlines.

        Yaw is measured in the world XY-plane (counter-clockwise from +X).
        The facing direction is the cross-product of (right→left) vectors
        projected onto the horizontal plane.
        """
        T = coco17.shape[0]

        # COCO indices: L_shoulder=5, R_shoulder=6, L_hip=11, R_hip=12
        l_sh = coco17[:, 5, :2]  # (T, 2)
        r_sh = coco17[:, 6, :2]
        l_hp = coco17[:, 11, :2]
        r_hp = coco17[:, 12, :2]

        # right→left vector (average of shoulder & hip)
        rl_sh = l_sh - r_sh  # (T, 2)
        rl_hp = l_hp - r_hp
        rl = 0.5 * (rl_sh + rl_hp)

        # Facing direction = perpendicular to right→left (rotate -90°)
        facing_x = rl[:, 1]   #  rl_y
        facing_y = -rl[:, 0]  # -rl_x

        yaw: np.ndarray = np.arctan2(facing_y, facing_x).astype(np.float32)
        return yaw
