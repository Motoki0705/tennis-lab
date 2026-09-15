"""AMASS/ACCAD SMPL-H adapter for the canonical COCO-17 motion contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip
from src.tasks.plcs.motion.contracts import Coco17MotionClip, MotionSourceKind
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix


class AccadCoco17Adapter:
    """Evaluate SMPL-H and regress the same COCO-17 semantics used by GVHMR."""

    def __init__(
        self,
        *,
        smplh_model_path: str | Path,
        coco17_regressor_path: str | Path,
        device: str | torch.device,
    ) -> None:
        self.smplh_model_path = Path(smplh_model_path)
        self.coco17_regressor_path = Path(coco17_regressor_path)
        self.device = torch.device(device)
        self._models: dict[str, Any] = {}
        self._regressor: torch.Tensor | None = None

    def _model(self, gender: str) -> Any:
        if gender not in self._models:
            import smplx  # type: ignore[import-untyped]

            model = smplx.create(
                model_path=str(self.smplh_model_path.parent),
                model_type="smplh",
                gender=gender,
                num_betas=16,
                use_pca=False,
                ext="pkl",
            )
            self._models[gender] = model.to(self.device).eval()
        return self._models[gender]

    def _coco17_regressor(self) -> torch.Tensor:
        if self._regressor is None:
            if not self.coco17_regressor_path.is_file():
                raise FileNotFoundError(
                    f"SMPL COCO-17 regressor not found: {self.coco17_regressor_path}"
                )
            value = torch.load(
                self.coco17_regressor_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(value, torch.Tensor):
                raise TypeError("SMPL COCO-17 regressor asset must contain a tensor.")
            if value.is_sparse:
                value = value.to_dense()
            if value.shape != (17, 6890):
                raise ValueError(
                    "SMPL COCO-17 regressor must have shape (17,6890), "
                    f"got {tuple(value.shape)}."
                )
            self._regressor = value.to(device=self.device, dtype=torch.float32)
        return self._regressor

    def convert(
        self,
        source: PLCSMotionClip,
        *,
        batch_size: int = 64,
    ) -> Coco17MotionClip:
        """Convert every source frame without temporal resampling."""
        if not isinstance(source, PLCSMotionClip):
            raise TypeError("source must be PLCSMotionClip.")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        model = self._model(source.gender)
        regressor = self._coco17_regressor()
        betas = np.asarray(source.betas[: int(model.num_betas)], dtype=np.float32)
        joints: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, source.frame_count, batch_size):
                end = min(start + batch_size, source.frame_count)
                count = end - start
                output = cast(Any, model)(
                    betas=torch.from_numpy(betas[None])
                    .to(self.device)
                    .repeat(count, 1),
                    global_orient=torch.from_numpy(
                        np.asarray(
                            source.global_orient_axis_angle[start:end], dtype=np.float32
                        )
                    ).to(self.device),
                    body_pose=torch.from_numpy(
                        np.asarray(
                            source.body_pose_axis_angle[start:end], dtype=np.float32
                        )
                    ).to(self.device),
                    left_hand_pose=torch.from_numpy(
                        np.asarray(
                            source.left_hand_pose_axis_angle[start:end],
                            dtype=np.float32,
                        )
                    ).to(self.device),
                    right_hand_pose=torch.from_numpy(
                        np.asarray(
                            source.right_hand_pose_axis_angle[start:end],
                            dtype=np.float32,
                        )
                    ).to(self.device),
                    transl=torch.from_numpy(
                        np.asarray(
                            source.root_translation_m[start:end], dtype=np.float32
                        )
                    ).to(self.device),
                    return_verts=True,
                )
                vertices = output.vertices
                if not isinstance(vertices, torch.Tensor) or vertices.shape != (
                    count,
                    6890,
                    3,
                ):
                    raise RuntimeError(
                        "SMPL-H adapter expected vertices shaped "
                        f"({count},6890,3), got {getattr(vertices, 'shape', None)}."
                    )
                joints.append(torch.einsum("jv,bvc->bjc", regressor, vertices).cpu())

        root_axis_angle = torch.from_numpy(
            np.asarray(source.global_orient_axis_angle, dtype=np.float32)
        )
        return Coco17MotionClip(
            source_id=Path(source.source_path).stem,
            source_path=source.source_path,
            source_kind=MotionSourceKind.ACCAD,
            category=source.category.value,
            gender=source.gender,
            fps=source.fps,
            timestamps_s=np.arange(source.frame_count, dtype=np.float64) / source.fps,
            joints_3d_m=np.asarray(torch.cat(joints).numpy(), dtype=np.float32),
            root_translation_m=np.asarray(source.root_translation_m, dtype=np.float32),
            root_rotation=np.asarray(
                axis_angle_to_matrix(root_axis_angle).numpy(), dtype=np.float32
            ),
            joint_confidence=np.ones((source.frame_count, 17), dtype=np.float32),
            frame_valid=np.ones(source.frame_count, dtype=np.bool_),
            provenance={
                "adapter": "amass_smplh_to_coco17_v1",
                "source": source.metadata(),
            },
        )


__all__ = ["AccadCoco17Adapter"]
