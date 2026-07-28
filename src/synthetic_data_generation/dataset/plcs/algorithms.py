"""Configuration-selectable PLCS avatar-control and motion algorithms."""

from __future__ import annotations

from src.synthetic_data_generation.dataset.algorithms import (
    AlgorithmDefinition,
    AlgorithmRegistry,
)

AVATAR_CONTROL_ALGORITHMS = AlgorithmRegistry(
    namespace="plcs.avatar_control",
    definitions=(
        AlgorithmDefinition(
            name="gaussianavatar_query_lbs",
            implementation="gaussianavatar_query_lbs",
            description="Control each Gaussian with interpolated SMPL-X LBS weights.",
        ),
        AlgorithmDefinition(
            name="hugs_topk_lbs",
            implementation="hugs_topk_lbs",
            description="Blend top-k neighboring SMPL-X vertex transforms per Gaussian.",
        ),
    ),
)

MOTION_ALGORITHMS = AlgorithmRegistry(
    namespace="plcs.motion",
    definitions=(
        AlgorithmDefinition(
            name="seeded_court_motion",
            implementation="seeded_court_motion",
            description=(
                "Generate deterministic single/multi-person court positions, yaw, "
                "and pose schedules."
            ),
        ),
    ),
)

__all__ = ["AVATAR_CONTROL_ALGORITHMS", "MOTION_ALGORITHMS"]
