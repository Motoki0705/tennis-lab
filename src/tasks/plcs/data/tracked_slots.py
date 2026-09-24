"""Camera-local 2D track IDs own slots for the entire input scene."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrackedSlots:
    keypoints: Tensor
    visibility: Tensor
    local_track_ids: Tensor  # (V,P), -1 never allocated
    detection_indices: Tensor  # (V,T,P), -1 absent


class FixedTrackRegistry:
    """No expiry/reuse. Keep this registry when processing pieces of one scene."""

    def __init__(self, camera_ids: tuple[str, ...], *, num_slots: int = 4) -> None:
        if not camera_ids or len(set(camera_ids)) != len(camera_ids) or any(not camera for camera in camera_ids):
            raise ValueError("Camera IDs must be nonempty and unique")
        if type(num_slots) is not int or num_slots < 1:
            raise ValueError("num_slots must be a positive integer")
        self.camera_ids, self.num_slots = camera_ids, num_slots
        self.assignments: list[dict[int, int]] = [{} for _ in camera_ids]

    def pack(self, keypoints: Tensor, visibility: Tensor, track_ids: Tensor) -> TrackedSlots:
        if keypoints.ndim != 5 or keypoints.shape[-2:] != (17, 2) or visibility.shape != keypoints.shape[:-1]:
            raise ValueError("Tracked poses require (V,T,D,17,2) and joint visibility")
        v, t, carriers = keypoints.shape[:3]
        if v != len(self.camera_ids) or visibility.dtype != torch.bool or track_ids.dtype != torch.int64:
            raise ValueError("Invalid camera count, visibility or track ID dtype")
        if any(value.device.type != "cpu" for value in (keypoints, visibility, track_ids)):
            raise ValueError("Track slot packing is a CPU input operation")
        if track_ids.shape == (v, carriers):
            track_ids = track_ids[:, None].expand(v, t, carriers)
        if track_ids.shape != (v, t, carriers) or bool((track_ids < -1).any()):
            raise ValueError("Local track IDs require (V,D) or (V,T,D); missing=-1")
        if bool((visibility.any(-1) & track_ids.lt(0)).any()):
            raise ValueError("Every observed person must have a local track ID")
        # Validate capacity for every view before mutating persistent assignments.
        pending = []
        for view in range(v):
            ids = sorted(int(x) for x in track_ids[view].unique().tolist() if x >= 0)
            new = [identity for identity in ids if identity not in self.assignments[view]]
            if len(self.assignments[view]) + len(new) > self.num_slots:
                raise ValueError(f"Camera {self.camera_ids[view]} exceeds cumulative track capacity {self.num_slots}; slots are never reused")
            for identity in ids:
                if bool(((track_ids[view] == identity).sum(-1) > 1).any()):
                    raise ValueError("A camera-local ID occupies multiple detections in one frame")
            pending.append(new)
        packed = keypoints.new_zeros(v, t, self.num_slots, 17, 2)
        visible = torch.zeros(v, t, self.num_slots, 17, dtype=torch.bool)
        indices = torch.full((v, t, self.num_slots), -1, dtype=torch.int64)
        local_ids = torch.full((v, self.num_slots), -1, dtype=torch.int64)
        for view, new in enumerate(pending):
            mapping = self.assignments[view]
            for identity in new:
                mapping[identity] = len(mapping)
            for identity, slot in mapping.items():
                local_ids[view, slot] = identity
                frames, detection = (track_ids[view] == identity).nonzero(as_tuple=True)
                packed[view, frames, slot] = keypoints[view, frames, detection]
                visible[view, frames, slot] = visibility[view, frames, detection]
                indices[view, frames, slot] = detection
        packed = packed.masked_fill(~visible[..., None], 0)
        return TrackedSlots(packed, visible, local_ids, indices)
