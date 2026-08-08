"""Typed trajectory generation, coverage selection, and 3-D sampling."""

from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    assign_court_targets_for_groups,
    assign_group_disjoint_splits,
    assign_group_shards,
    build_court_dataset_plan,
    select_budgeted_coverage,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
)

__all__ = [
    "assign_court_targets_for_groups",
    "assign_group_disjoint_splits",
    "assign_group_shards",
    "build_court_dataset_plan",
    "derive_orbit_centers",
    "generate_trajectory_candidates",
    "sample_uniform_arc_length",
    "select_budgeted_coverage",
]
