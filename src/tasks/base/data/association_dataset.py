"""Shared sampling and labeling mechanics for 2D-only association datasets."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.observation_tracking import (
    ObservationTrackingConfig,
    track_multiview_observations,
)
from src.tasks.base.data.rng import require_run_seed
from src.tasks.base.data.scene_dataset import (
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
)
from src.tasks.base.data.track_query_reference import (
    include_evaluation_reference_camera,
    resolve_evaluation_reference_camera_id,
    select_seeded_training_reference_camera_id,
)
from src.tasks.base.generate_dataset import (
    CourtViewRecord,
    resolve_court_keypoint_contract,
)
from src.utils.data.camera_sampling import camera_candidate_indices


class AssociationSceneDataset(SceneDatasetBase[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: Any,
        seed: int | None = None,
        augment: bool = False,
        reference_camera_id: str | None = None,
    ) -> None:
        self.augment = augment
        self.hydra_cfg = config
        self.reference_camera_id = reference_camera_id
        self.num_slots = int(config.model.num_slots)
        self.max_identities = int(config.model.max_identities)
        self.court_contract = resolve_court_keypoint_contract(
            str(config.court_keypoints.selector)
        )
        if self.court_contract.selector != "camera_view_v2":
            raise ValueError(
                "Association datasets require camera-local V2 observations"
            )
        data = config.data
        self.observation_tracking_config = ObservationTrackingConfig.from_mapping(
            data.association
        )
        self.court_views = self.load_court_views(Path(scene_dir), Path(split_file))
        self._cache: dict[int, dict[str, Tensor]] = {}
        self.cache_size = int(data.validation_cache_size) if not augment else 0
        super().__init__(
            config=SceneDatasetConfig(
                scene_dir=Path(scene_dir),
                split_file=Path(split_file),
                seq_len_range=tuple(data.seq_len_range),
                num_views_range=tuple(data.num_views_range),
                camera_mode=data.camera_mode,
                crop_mode="random" if augment else "center",
                min_num_frames=1,
                min_num_cameras=1,
                camera_candidates=camera_candidate_indices(data.camera_candidates),
            ),
            seed=require_run_seed(config) if seed is None else seed,
            sample_local_rng=not augment,
        )

    @abstractmethod
    def load_court_views(
        self, root: Path, split: Path
    ) -> dict[str, tuple[CourtViewRecord, ...]]: ...

    def selected_views(
        self, scene: Scene
    ) -> tuple[tuple[int, ...], tuple[CourtViewRecord, ...], int]:
        complete = self.court_views[scene.path.name]
        indices = self.select_cameras(scene).indices
        if not self.augment:
            indices = include_evaluation_reference_camera(
                tuple(v.camera_id for v in complete),
                indices,
                requested_camera_id=self.reference_camera_id,
                candidate_camera_indices=self.config.camera_candidates,
                rng=self.rng,
            )
        views = tuple(complete[i] for i in indices)
        ids = tuple(v.camera_id for v in views)
        reference = (
            select_seeded_training_reference_camera_id(ids, rng=self.rng)
            if self.augment
            else resolve_evaluation_reference_camera_id(
                ids, requested_camera_id=self.reference_camera_id
            )
        )
        return indices, views, ids.index(reference)

    def finish_sample(
        self,
        *,
        uv: Tensor,
        visible: Tensor,
        court: Tensor,
        court_visible: Tensor,
        identity: Tensor,
        camera_indices: tuple[int, ...],
        views: tuple[CourtViewRecord, ...],
        reference: int,
    ) -> dict[str, Tensor]:
        tracked = track_multiview_observations(
            uv,
            visible,
            num_slots=self.num_slots,
            config=self.observation_tracking_config,
            camera_indices=camera_indices,
            debug_provenance=identity,
        )
        if tracked.debug_provenance is None:
            raise RuntimeError("Association tracker lost teacher provenance")
        # Label creation follows observation-only local tracking. No GT fields enter forward.
        teacher = tracked.debug_provenance
        observed_ids = teacher[tracked.visibility.any(-1) & teacher.ge(0)].unique()
        if len(observed_ids) > self.max_identities:
            raise ValueError("Clip identity count exceeds model.max_identities")
        compact = torch.full_like(teacher, -1)
        for index, physical_id in enumerate(observed_ids):
            compact[teacher == physical_id] = index
        absolute = torch.tensor(
            [v.camera_center_court_m[1] > 0 for v in views], dtype=torch.bool
        )
        return {
            "object_uv": tracked.values,
            "object_vis": tracked.visibility,
            "court_kp": court.masked_fill(~court_visible[..., None], 0),
            "court_vis": court_visible,
            "padding_mask": torch.zeros(uv.shape[:2], dtype=torch.bool),
            "reference_view_index": torch.tensor(reference, dtype=torch.long),
            "object_id_target": compact,
            "side_target": absolute ^ absolute[reference],
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index in self._cache:
            return self._cache[index]
        sample: dict[str, Tensor] = super().__getitem__(index)
        if self.cache_size:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = sample
        return sample
