"""Base data abstractions shared across tasks."""

from src.tasks.base.data.canonical_tracking import permute_tracking_views
from src.tasks.base.data.chunk_manager import (
    ChunkGenerator,
    ChunkInfo,
    ChunkManager,
    ChunkState,
)
from src.tasks.base.data.court_peaks import (
    COURT_PHYSICAL_INDICES_BY_CLASS,
    COURT_SEMANTIC_CLASS_NAMES,
    CourtObservationProfile,
    CourtPeakBatch,
    CourtPeakFrame,
    assemble_court_peak_batch,
    court_peak_batch_from_model_input,
    ordered_court_to_semantic_peaks,
    parse_court_observation_profile,
    predicted_peaks_to_normalized,
    reference_context_validity,
    reference_view_mask,
)
from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.base.data.reference_orientation import (
    camera_centers_from_scene_payload,
    deterministic_sample_rng,
    orientation_signs_from_camera_centers,
    reflect_court_vectors,
    reflect_heading,
    select_counterfactual_reference_views,
    select_reference_view,
    validate_declared_reference_orientation,
)
from src.tasks.base.data.scene_dataset import (
    CameraSelection,
    Scene,
    SceneDataContractError,
    SceneDatasetBase,
    SceneDatasetConfig,
    SceneHeader,
    TemporalWindow,
)

__all__ = [
    "ChunkGenerator",
    "ChunkInfo",
    "ChunkManager",
    "ChunkState",
    "BaseDatasetWriter",
    "CameraSelection",
    "camera_centers_from_scene_payload",
    "deterministic_sample_rng",
    "Scene",
    "SceneDataContractError",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneDatasetConfig",
    "TemporalWindow",
    "COURT_PHYSICAL_INDICES_BY_CLASS",
    "COURT_SEMANTIC_CLASS_NAMES",
    "CourtObservationProfile",
    "CourtPeakBatch",
    "CourtPeakFrame",
    "assemble_court_peak_batch",
    "court_peak_batch_from_model_input",
    "ordered_court_to_semantic_peaks",
    "parse_court_observation_profile",
    "permute_tracking_views",
    "predicted_peaks_to_normalized",
    "orientation_signs_from_camera_centers",
    "reference_context_validity",
    "reference_view_mask",
    "reflect_court_vectors",
    "reflect_heading",
    "select_counterfactual_reference_views",
    "select_reference_view",
    "validate_declared_reference_orientation",
]
