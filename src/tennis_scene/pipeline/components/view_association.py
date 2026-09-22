"""Learned camera side and cross-view IDs ahead of geometric reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.association.inference import AssociationPredictor
from src.tasks.base.data.track_query_reference import ReferenceCameraSelection
from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tennis_scene.pipeline.court_reference import (
    CourtReferenceContext,
    CourtReferenceRuntimeConfig,
    prepare_court_reference,
)


@dataclass(frozen=True)
class ViewAssociationResult:
    camera_ids: tuple[str, ...]
    reference_camera_id: str
    view_half_turns: tuple[bool, ...]
    object_ids: NDArray[np.int64]  # (V,T,P); -1 = missing or non-target
    side_logits: NDArray[np.float32]

    def geometry_context(
        self,
        court_kp: NDArray[np.float32],
        court_vis: NDArray[np.bool_],
        *,
        size: tuple[int, int],
        frame_index: int = 0,
    ) -> CourtReferenceContext:
        """Calibrate only AFTER inferred side information is available."""
        return prepare_court_reference(
            camera_ids=self.camera_ids,
            keypoints=court_kp,
            visibility=court_vis,
            contract=resolve_court_keypoint_contract("camera_view_v2"),
            config=CourtReferenceRuntimeConfig(
                reference_camera=self.reference_camera_id,
                view_half_turns=self.view_half_turns,
            ),
            size=size,
            frame_index=frame_index,
        )

    def group_observations(
        self, uv: NDArray[np.float32], visibility: NDArray[np.bool_]
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.bool_]]:
        """Gather identical clip IDs into (ID,V,T,J,2) for downstream triangulation."""
        if uv.shape[:-2] != self.object_ids.shape or visibility.shape != uv.shape[:-1]:
            raise ValueError("Association and observations must share (V,T,P) axes")
        identities = np.unique(self.object_ids[self.object_ids >= 0])
        views, frames, _, joints, _ = uv.shape
        grouped: NDArray[np.float32] = np.zeros(
            (len(identities), views, frames, joints, 2), dtype=np.float32
        )
        visible: NDArray[np.bool_] = np.zeros(grouped.shape[:-1], dtype=np.bool_)
        for row, identity in enumerate(identities):
            selected = self.object_ids == identity
            if (selected.sum(-1) > 1).any():
                raise ValueError(
                    "One physical identity cannot occupy multiple slots in the same view/frame"
                )
            for v, t, slot in np.argwhere(selected):
                grouped[row, v, t] = uv[v, t, slot]
                visible[row, v, t] = visibility[v, t, slot]
        return identities, grouped, visible


class ViewAssociationModule:
    """Inference on unassociated camera-local player/ball observations."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        self.predictor = AssociationPredictor.load(checkpoint, device=device)

    def process(
        self,
        *,
        camera_ids: tuple[str, ...],
        reference_camera_id: str,
        object_uv: NDArray[np.float32],
        object_vis: NDArray[np.bool_],
        court_kp: NDArray[np.float32],
        court_vis: NDArray[np.bool_],
        padding_mask: NDArray[np.bool_],
    ) -> ViewAssociationResult:
        selection = ReferenceCameraSelection(camera_ids, reference_camera_id)
        if object_uv.shape[0] != len(camera_ids):
            raise ValueError("Camera IDs must match the observation view axis")
        output = self.predictor.predict(
            {
                "object_uv": torch.from_numpy(object_uv).unsqueeze(0),
                "object_vis": torch.from_numpy(object_vis).unsqueeze(0),
                "court_kp": torch.from_numpy(court_kp).unsqueeze(0),
                "court_vis": torch.from_numpy(court_vis).unsqueeze(0),
                "padding_mask": torch.from_numpy(padding_mask).unsqueeze(0),
                "reference_view_index": selection.as_tensor(),
            }
        )
        return ViewAssociationResult(
            camera_ids=camera_ids,
            reference_camera_id=reference_camera_id,
            view_half_turns=tuple(
                bool(x) for x in output["view_half_turns"][0].tolist()
            ),
            object_ids=output["object_ids"][0].numpy(),
            side_logits=output["side_logits"][0].float().numpy(),
        )
