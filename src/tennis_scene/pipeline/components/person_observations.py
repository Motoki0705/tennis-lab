"""Pipeline component for raw, unassociated camera-local player observations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.tennis_scene.pipeline.model_io.observations import ObjectObservations
from src.tennis_scene.pipeline.model_io.people import PersonObservationAdapter
from src.utils.video import VideoInfo


class PersonObservationModule:
    def __init__(self, adapter: PersonObservationAdapter) -> None:
        self.adapter = adapter

    def process(
        self,
        video_paths: Sequence[Path],
        video_infos: Sequence[VideoInfo],
        camera_ids: tuple[str, ...],
        *,
        num_frames: int,
        polygons: Sequence[tuple[tuple[float, float], ...] | None],
    ) -> ObjectObservations:
        return self.adapter.observe(video_paths, video_infos, camera_ids, num_frames=num_frames, polygons=polygons)
