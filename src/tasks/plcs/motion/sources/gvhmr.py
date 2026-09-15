"""GVHMR global-SMPL adapter for the canonical COCO-17 motion contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import SmplCoco17Reconstructor
from src.tasks.plcs.motion.contracts import Coco17MotionClip, MotionSourceKind
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix

# GVHMR emits an aligned-Y-up world. Rotating +90 degrees around X maps it to
# the PLCS right-handed Z-up convention: (x, y, z) -> (x, -z, y).
GVHMR_Y_UP_TO_PLCS_Z_UP: NDArray[np.float32] = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


class GvhmrCoco17Adapter:
    """Convert GVHMR global output while preserving its trajectory and root pose."""

    _WIDTHS = {
        "body_pose": 63,
        "betas": 10,
        "global_orient": 3,
        "transl": 3,
    }

    def __init__(self, joint_reconstructor: SmplCoco17Reconstructor) -> None:
        if not isinstance(joint_reconstructor, SmplCoco17Reconstructor):
            raise TypeError("joint_reconstructor must be SmplCoco17Reconstructor.")
        self.joint_reconstructor = joint_reconstructor

    def convert(
        self,
        global_parameters: Mapping[str, torch.Tensor],
        *,
        source_id: str,
        source_path: str | Path,
        fps: float,
        joint_confidence: NDArray[np.float32],
        frame_valid: NDArray[np.bool_],
        provenance: Mapping[str, object],
    ) -> Coco17MotionClip:
        """Convert one tracked player without smoothing or temporal resampling."""
        if set(global_parameters) != set(self._WIDTHS):
            raise ValueError(
                f"GVHMR global parameters must contain exactly {sorted(self._WIDTHS)}."
            )
        params = {name: global_parameters[name] for name in self._WIDTHS}
        joints_y_up = self.joint_reconstructor.reconstruct(params)
        frame_count = int(joints_y_up.shape[0])
        basis = torch.from_numpy(GVHMR_Y_UP_TO_PLCS_Z_UP)
        joints_z_up = torch.einsum("ij,tkj->tki", basis, joints_y_up)
        translation_y_up = params["transl"].detach().float().cpu()
        translation_z_up = torch.einsum("ij,tj->ti", basis, translation_y_up)
        rotation_y_up = axis_angle_to_matrix(
            params["global_orient"].detach().float().cpu()
        )
        # ``global_orient`` actively maps the unchanged SMPL rest frame into
        # world coordinates.  Converting that world (and the resulting joints)
        # therefore left-multiplies by the world-frame basis transform.  A
        # similarity transform would incorrectly rotate the SMPL rest frame too.
        rotation_z_up = torch.einsum("ij,tjk->tik", basis, rotation_y_up)

        confidence = np.asarray(joint_confidence)
        valid = np.asarray(frame_valid)
        if confidence.shape != (frame_count, 17):
            raise ValueError(
                f"joint_confidence must have shape ({frame_count},17), "
                f"got {confidence.shape}."
            )
        if valid.shape != (frame_count,):
            raise ValueError(
                f"frame_valid must have shape ({frame_count},), got {valid.shape}."
            )
        source = Path(source_path).resolve()
        combined_provenance = dict(provenance)
        combined_provenance["adapter"] = "gvhmr_global_y_up_to_coco17_z_up_v1"
        return Coco17MotionClip(
            source_id=source_id,
            source_path=str(source),
            source_kind=MotionSourceKind.GVHMR,
            category="tennis",
            gender="neutral",
            fps=fps,
            timestamps_s=np.arange(frame_count, dtype=np.float64) / float(fps),
            joints_3d_m=np.asarray(joints_z_up.numpy(), dtype=np.float32),
            root_translation_m=np.asarray(translation_z_up.numpy(), dtype=np.float32),
            root_rotation=np.asarray(rotation_z_up.numpy(), dtype=np.float32),
            joint_confidence=confidence,
            frame_valid=valid,
            provenance=combined_provenance,
        )


__all__ = ["GVHMR_Y_UP_TO_PLCS_Z_UP", "GvhmrCoco17Adapter"]
