"""Stable public boundary for complete BLCS physics source trajectories.

The task-local scene generators remain implementation details.  Consumers get
validated NumPy data, complete lifecycle mappings, and explicit proposal
diagnostics instead of depending on mutable generator scene objects.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    BLCSSceneGenerator,
    GeneratorConfig,
)
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    is_retryable_full_physics_rejection,
)

BLCSGeneratorConfiguration: TypeAlias = GeneratorConfig

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TIMELINE_KEYS = {
    "num_frames",
    "min_tracks",
    "max_tracks",
    "max_concurrent",
    "min_reuse_gap_frames",
    "start_index_range",
    "min_active_frames",
    "overlap_probability",
    "min_gap_frames",
    "max_gap_frames",
}
_GENERATOR_KEYS = {"physics", "rally", "camera", "targeted_velocity", "court"}
_PHYSICS_KEYS = {
    "gravity",
    "k_drag",
    "k_magnus",
    "e_z",
    "mu",
    "alpha_net",
    "alpha_net_cord",
    "alpha_fence",
    "net_half_thickness",
    "net_cord_radius",
    "dt",
    "use_drag",
    "use_magnus",
    "wind",
    "gravity_range",
    "k_drag_range",
    "k_magnus_range",
    "e_z_range",
    "mu_range",
    "wind_speed_range",
    "wind_direction_range_deg",
}
_RALLY_KEYS = {
    "z_range",
    "spin_x_range",
    "spin_y_range",
    "spin_z_range",
    "max_sim_frames",
    "output_fps",
    "sim_fps",
    "max_rallies",
    "max_total_frames",
    "hit_timing_range",
    "return_z_range",
    "serve_probability",
    "serve_z_range",
    "toss_vz_range",
    "toss_xy_noise_range",
    "toss_max_frames",
    "toss_z0_tolerance",
    "volley_probability",
    "normal_return_probability",
    "late_return_probability",
    "out_court_target_probability",
}
_CAMERA_KEYS = {
    "z_min",
    "z_max",
    "hfov_deg",
    "image_size",
    "fixed_look_at",
    "fixed_baseline_clear_extra",
    "fixed_position_noise_radius",
    "fixed_look_at_xy_radius",
    "layout",
    "broadcast_setback",
    "broadcast_height",
    "broadcast_hfov_deg",
    "broadcast_look_at_y",
    "broadcast_look_at_height",
    "broadcast_position_noise_radius",
    "broadcast_look_at_xy_radius",
    "broadcast_hfov_jitter_deg",
    "broadcast_setback_range",
    "broadcast_height_range",
    "broadcast_court_width_frac_range",
}
_TARGETED_VELOCITY_KEYS = {
    "drive_elevation_range_deg",
    "lob_elevation_range_deg",
    "lob_probability",
    "max_ballistic_apex_height_m",
    "gravity",
    "net_elevation_step_deg",
    "landing_refine_enabled",
    "landing_refine_max_iters",
    "landing_refine_tolerance_m",
    "landing_sim_max_frames",
    "target_margin_m",
}
_COURT_KEYS = {"net_post_offset_x", "net_post_offset_x_range"}
class BLCSPhysicsProposalRejected(RuntimeError):
    """Signal that one stochastic physics proposal may be resampled."""


class BLCSPhysicsProposalExhausted(RuntimeError):
    """Raised when a source object has no valid proposal within its budget."""

    def __init__(self, diagnostic: BLCSProposalDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            "BLCS physics proposals exhausted "
            f"{diagnostic.maximum_attempts} attempts for "
            f"{diagnostic.source_trajectory_id!r}; "
            f"rejections={list(diagnostic.rejected_attempts)!r}."
        )


@dataclass(frozen=True, slots=True)
class BLCSTimelineSpec:
    """Stable task-owned lifecycle specification for one source scene."""

    num_frames: int
    min_tracks: int
    max_tracks: int
    max_concurrent: int
    min_reuse_gap_frames: int
    start_index_range: tuple[int, int]
    min_active_frames: int
    overlap_probability: float
    min_gap_frames: int
    max_gap_frames: int

    def __post_init__(self) -> None:
        _positive_int(self.num_frames, name="num_frames")
        _positive_int(self.min_tracks, name="min_tracks")
        _positive_int(self.max_tracks, name="max_tracks")
        if self.min_tracks < 2:
            raise ValueError("BLCS physics source scenes require at least two tracks.")
        if self.max_tracks < self.min_tracks:
            raise ValueError("max_tracks must be greater than or equal to min_tracks.")
        _positive_int(self.max_concurrent, name="max_concurrent")
        if self.max_concurrent > self.max_tracks:
            raise ValueError("max_concurrent must not exceed max_tracks.")
        _non_negative_int(
            self.min_reuse_gap_frames,
            name="min_reuse_gap_frames",
        )
        start_range = _integer_pair(
            self.start_index_range,
            name="start_index_range",
        )
        if start_range[0] > start_range[1]:
            raise ValueError("start_index_range must be increasing.")
        _positive_int(self.min_active_frames, name="min_active_frames")
        if self.min_active_frames > self.num_frames:
            raise ValueError("min_active_frames must not exceed num_frames.")
        probability = _finite_float(
            self.overlap_probability,
            name="overlap_probability",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("overlap_probability must be in [0, 1].")
        _non_negative_int(self.min_gap_frames, name="min_gap_frames")
        _non_negative_int(self.max_gap_frames, name="max_gap_frames")
        if self.max_gap_frames < self.min_gap_frames:
            raise ValueError("max_gap_frames must not be less than min_gap_frames.")
        object.__setattr__(self, "start_index_range", start_range)
        object.__setattr__(self, "overlap_probability", probability)

    @classmethod
    def from_mapping(cls, value: object) -> BLCSTimelineSpec:
        """Parse a strict mapping without inheriting generator defaults."""
        raw = _exact_mapping(value, keys=_TIMELINE_KEYS, name="timeline")
        return cls(
            num_frames=_mapping_int(raw, "num_frames"),
            min_tracks=_mapping_int(raw, "min_tracks"),
            max_tracks=_mapping_int(raw, "max_tracks"),
            max_concurrent=_mapping_int(raw, "max_concurrent"),
            min_reuse_gap_frames=_mapping_int(raw, "min_reuse_gap_frames"),
            start_index_range=_integer_pair(
                raw["start_index_range"],
                name="timeline.start_index_range",
            ),
            min_active_frames=_mapping_int(raw, "min_active_frames"),
            overlap_probability=_finite_float(
                raw["overlap_probability"],
                name="timeline.overlap_probability",
            ),
            min_gap_frames=_mapping_int(raw, "min_gap_frames"),
            max_gap_frames=_mapping_int(raw, "max_gap_frames"),
        )

    def _to_internal(self) -> TimelineConfig:
        return TimelineConfig(
            num_frames=self.num_frames,
            min_tracks=self.min_tracks,
            max_tracks=self.max_tracks,
            max_concurrent=self.max_concurrent,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            start_index_range=self.start_index_range,
            min_active_frames=self.min_active_frames,
            overlap_probability=self.overlap_probability,
            min_gap_frames=self.min_gap_frames,
            max_gap_frames=self.max_gap_frames,
        )


@dataclass(frozen=True, slots=True)
class BLCSPhysicsSourceSettings:
    """Explicit settings for deterministic bounded source generation."""

    timeline: BLCSTimelineSpec
    maximum_physics_attempts_per_object: int
    device: str

    def __post_init__(self) -> None:
        if not isinstance(self.timeline, BLCSTimelineSpec):
            raise TypeError("timeline must be a BLCSTimelineSpec.")
        _positive_int(
            self.maximum_physics_attempts_per_object,
            name="maximum_physics_attempts_per_object",
        )
        if not isinstance(self.device, str) or not self.device.strip():
            raise TypeError("device must be an explicit non-empty string.")
        device = torch.device(self.device)
        if device.type != "cpu":
            raise ValueError("BLCS physics source generation requires CPU execution.")
        object.__setattr__(self, "device", str(device))

    @classmethod
    def from_mapping(cls, value: object) -> BLCSPhysicsSourceSettings:
        """Parse strict source settings with no implicit fallback values."""
        raw = _exact_mapping(
            value,
            keys={
                "timeline",
                "maximum_physics_attempts_per_object",
                "device",
            },
            name="BLCS physics source settings",
        )
        timeline_value = raw["timeline"]
        timeline = (
            timeline_value
            if isinstance(timeline_value, BLCSTimelineSpec)
            else BLCSTimelineSpec.from_mapping(timeline_value)
        )
        return cls(
            timeline=timeline,
            maximum_physics_attempts_per_object=_mapping_int(
                raw,
                "maximum_physics_attempts_per_object",
            ),
            device=_mapping_text(raw, "device"),
        )


def build_blcs_generator_configuration(
    value: object,
) -> BLCSGeneratorConfiguration:
    """Build the hidden task generator config from one strict resolved mapping.

    This is the public construction boundary for external orchestration code.
    Every field is required, unknown fields are rejected, and no task-local
    default is copied into the caller.
    """
    from src.tasks.blcs.generate_dataset.simulation.ball_physics import PhysicsConfig
    from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyConfig
    from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
        TargetedVelocityConfig,
    )
    from src.utils.projection.camera_projector import (
        SUPPORTED_LAYOUTS,
        CameraConfig,
    )
    from src.utils.schema.court import CourtConfig

    raw = _exact_mapping(
        value,
        keys=_GENERATOR_KEYS,
        name="BLCS generator configuration",
    )
    physics_raw = _mapping_section(
        raw,
        "physics",
        keys=_PHYSICS_KEYS,
    )
    rally_raw = _mapping_section(raw, "rally", keys=_RALLY_KEYS)
    camera_raw = _mapping_section(raw, "camera", keys=_CAMERA_KEYS)
    targeted_raw = _mapping_section(
        raw,
        "targeted_velocity",
        keys=_TARGETED_VELOCITY_KEYS,
    )
    court_raw = _mapping_section(raw, "court", keys=_COURT_KEYS)

    physics = PhysicsConfig(
        gravity=_positive_mapping_float(physics_raw, "gravity", section="physics"),
        k_drag=_non_negative_mapping_float(
            physics_raw,
            "k_drag",
            section="physics",
        ),
        k_magnus=_non_negative_mapping_float(
            physics_raw,
            "k_magnus",
            section="physics",
        ),
        e_z=_non_negative_mapping_float(physics_raw, "e_z", section="physics"),
        mu=_non_negative_mapping_float(physics_raw, "mu", section="physics"),
        alpha_net=_non_negative_mapping_float(
            physics_raw,
            "alpha_net",
            section="physics",
        ),
        alpha_net_cord=_non_negative_mapping_float(
            physics_raw,
            "alpha_net_cord",
            section="physics",
        ),
        alpha_fence=_non_negative_mapping_float(
            physics_raw,
            "alpha_fence",
            section="physics",
        ),
        net_half_thickness=_positive_mapping_float(
            physics_raw,
            "net_half_thickness",
            section="physics",
        ),
        net_cord_radius=_positive_mapping_float(
            physics_raw,
            "net_cord_radius",
            section="physics",
        ),
        dt=_positive_mapping_float(physics_raw, "dt", section="physics"),
        use_drag=_mapping_bool(physics_raw, "use_drag", section="physics"),
        use_magnus=_mapping_bool(physics_raw, "use_magnus", section="physics"),
        wind=_number_triple(physics_raw["wind"], name="physics.wind"),
        gravity_range=_optional_number_pair(
            physics_raw["gravity_range"],
            name="physics.gravity_range",
            minimum=0.0,
        ),
        k_drag_range=_optional_number_pair(
            physics_raw["k_drag_range"],
            name="physics.k_drag_range",
            minimum=0.0,
        ),
        k_magnus_range=_optional_number_pair(
            physics_raw["k_magnus_range"],
            name="physics.k_magnus_range",
            minimum=0.0,
        ),
        e_z_range=_optional_number_pair(
            physics_raw["e_z_range"],
            name="physics.e_z_range",
            minimum=0.0,
        ),
        mu_range=_optional_number_pair(
            physics_raw["mu_range"],
            name="physics.mu_range",
            minimum=0.0,
        ),
        wind_speed_range=_optional_number_pair(
            physics_raw["wind_speed_range"],
            name="physics.wind_speed_range",
            minimum=0.0,
        ),
        wind_direction_range_deg=_optional_number_pair(
            physics_raw["wind_direction_range_deg"],
            name="physics.wind_direction_range_deg",
        ),
    )

    output_fps = _positive_mapping_int(rally_raw, "output_fps", section="rally")
    sim_fps = _positive_mapping_int(rally_raw, "sim_fps", section="rally")
    if sim_fps < output_fps or sim_fps % output_fps != 0:
        raise ValueError(
            "rally.sim_fps must be an integer multiple of rally.output_fps."
        )
    return_probabilities = tuple(
        _probability(rally_raw[key], name=f"rally.{key}")
        for key in (
            "volley_probability",
            "normal_return_probability",
            "late_return_probability",
        )
    )
    if sum(return_probabilities) <= 0.0:
        raise ValueError("BLCS rally return probabilities must have positive mass.")
    rally = RallyConfig(
        z_range=_number_pair(rally_raw["z_range"], name="rally.z_range"),
        spin_x_range=_number_pair(
            rally_raw["spin_x_range"],
            name="rally.spin_x_range",
        ),
        spin_y_range=_number_pair(
            rally_raw["spin_y_range"],
            name="rally.spin_y_range",
        ),
        spin_z_range=_number_pair(
            rally_raw["spin_z_range"],
            name="rally.spin_z_range",
        ),
        max_sim_frames=_positive_mapping_int(
            rally_raw,
            "max_sim_frames",
            section="rally",
        ),
        output_fps=output_fps,
        sim_fps=sim_fps,
        max_rallies=_positive_mapping_int(
            rally_raw,
            "max_rallies",
            section="rally",
        ),
        max_total_frames=_positive_mapping_int(
            rally_raw,
            "max_total_frames",
            section="rally",
        ),
        hit_timing_range=_number_pair(
            rally_raw["hit_timing_range"],
            name="rally.hit_timing_range",
        ),
        return_z_range=_number_pair(
            rally_raw["return_z_range"],
            name="rally.return_z_range",
        ),
        serve_probability=_probability(
            rally_raw["serve_probability"],
            name="rally.serve_probability",
        ),
        serve_z_range=_number_pair(
            rally_raw["serve_z_range"],
            name="rally.serve_z_range",
        ),
        toss_vz_range=_number_pair(
            rally_raw["toss_vz_range"],
            name="rally.toss_vz_range",
        ),
        toss_xy_noise_range=_number_pair(
            rally_raw["toss_xy_noise_range"],
            name="rally.toss_xy_noise_range",
        ),
        toss_max_frames=_positive_mapping_int(
            rally_raw,
            "toss_max_frames",
            section="rally",
        ),
        toss_z0_tolerance=_non_negative_mapping_float(
            rally_raw,
            "toss_z0_tolerance",
            section="rally",
        ),
        volley_probability=return_probabilities[0],
        normal_return_probability=return_probabilities[1],
        late_return_probability=return_probabilities[2],
        out_court_target_probability=_probability(
            rally_raw["out_court_target_probability"],
            name="rally.out_court_target_probability",
        ),
    )

    layout = _mapping_text(camera_raw, "layout")
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            f"camera.layout must be one of {list(SUPPORTED_LAYOUTS)}, got {layout!r}."
        )
    z_min = _positive_mapping_float(camera_raw, "z_min", section="camera")
    z_max = _positive_mapping_float(camera_raw, "z_max", section="camera")
    if z_max < z_min:
        raise ValueError("camera.z_max must not be less than camera.z_min.")
    hfov = _field_of_view(camera_raw["hfov_deg"], name="camera.hfov_deg")
    broadcast_hfov = _field_of_view(
        camera_raw["broadcast_hfov_deg"],
        name="camera.broadcast_hfov_deg",
    )
    hfov_jitter = _non_negative_mapping_float(
        camera_raw,
        "broadcast_hfov_jitter_deg",
        section="camera",
    )
    if broadcast_hfov - hfov_jitter <= 0.0:
        raise ValueError("camera broadcast HFOV jitter permits a non-positive HFOV.")
    width_range = _optional_number_pair(
        camera_raw["broadcast_court_width_frac_range"],
        name="camera.broadcast_court_width_frac_range",
        minimum=0.0,
    )
    if width_range is not None:
        if not 0.0 < width_range[0] <= width_range[1] < 1.0:
            raise ValueError(
                "camera.broadcast_court_width_frac_range must lie inside (0, 1)."
            )
        if hfov_jitter > 0.0:
            raise ValueError(
                "camera broadcast width sampling and HFOV jitter are mutually exclusive."
            )
    camera = CameraConfig(
        z_min=z_min,
        z_max=z_max,
        hfov_deg=hfov,
        image_size=_positive_integer_pair(
            camera_raw["image_size"],
            name="camera.image_size",
        ),
        fixed_look_at=_number_triple(
            camera_raw["fixed_look_at"],
            name="camera.fixed_look_at",
        ),
        fixed_baseline_clear_extra=_non_negative_mapping_float(
            camera_raw,
            "fixed_baseline_clear_extra",
            section="camera",
        ),
        fixed_position_noise_radius=_non_negative_mapping_float(
            camera_raw,
            "fixed_position_noise_radius",
            section="camera",
        ),
        fixed_look_at_xy_radius=_non_negative_mapping_float(
            camera_raw,
            "fixed_look_at_xy_radius",
            section="camera",
        ),
        layout=layout,
        broadcast_setback=_positive_mapping_float(
            camera_raw,
            "broadcast_setback",
            section="camera",
        ),
        broadcast_height=_positive_mapping_float(
            camera_raw,
            "broadcast_height",
            section="camera",
        ),
        broadcast_hfov_deg=broadcast_hfov,
        broadcast_look_at_y=_mapping_float(
            camera_raw,
            "broadcast_look_at_y",
            section="camera",
        ),
        broadcast_look_at_height=_non_negative_mapping_float(
            camera_raw,
            "broadcast_look_at_height",
            section="camera",
        ),
        broadcast_position_noise_radius=_non_negative_mapping_float(
            camera_raw,
            "broadcast_position_noise_radius",
            section="camera",
        ),
        broadcast_look_at_xy_radius=_non_negative_mapping_float(
            camera_raw,
            "broadcast_look_at_xy_radius",
            section="camera",
        ),
        broadcast_hfov_jitter_deg=hfov_jitter,
        broadcast_setback_range=_optional_number_pair(
            camera_raw["broadcast_setback_range"],
            name="camera.broadcast_setback_range",
            minimum=0.0,
        ),
        broadcast_height_range=_optional_number_pair(
            camera_raw["broadcast_height_range"],
            name="camera.broadcast_height_range",
            minimum=0.0,
        ),
        broadcast_court_width_frac_range=width_range,
    )

    targeted_gravity = _positive_mapping_float(
        targeted_raw,
        "gravity",
        section="targeted_velocity",
    )
    if not math.isclose(targeted_gravity, physics.gravity, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "targeted_velocity.gravity must match physics.gravity exactly."
        )
    targeted = TargetedVelocityConfig(
        drive_elevation_range_deg=_number_pair(
            targeted_raw["drive_elevation_range_deg"],
            name="targeted_velocity.drive_elevation_range_deg",
        ),
        lob_elevation_range_deg=_number_pair(
            targeted_raw["lob_elevation_range_deg"],
            name="targeted_velocity.lob_elevation_range_deg",
        ),
        lob_probability=_probability(
            targeted_raw["lob_probability"],
            name="targeted_velocity.lob_probability",
        ),
        max_ballistic_apex_height_m=_positive_mapping_float(
            targeted_raw,
            "max_ballistic_apex_height_m",
            section="targeted_velocity",
        ),
        gravity=targeted_gravity,
        net_elevation_step_deg=_positive_mapping_float(
            targeted_raw,
            "net_elevation_step_deg",
            section="targeted_velocity",
        ),
        landing_refine_enabled=_mapping_bool(
            targeted_raw,
            "landing_refine_enabled",
            section="targeted_velocity",
        ),
        landing_refine_max_iters=_positive_mapping_int(
            targeted_raw,
            "landing_refine_max_iters",
            section="targeted_velocity",
        ),
        landing_refine_tolerance_m=_positive_mapping_float(
            targeted_raw,
            "landing_refine_tolerance_m",
            section="targeted_velocity",
        ),
        landing_sim_max_frames=_positive_mapping_int(
            targeted_raw,
            "landing_sim_max_frames",
            section="targeted_velocity",
        ),
        target_margin_m=_non_negative_mapping_float(
            targeted_raw,
            "target_margin_m",
            section="targeted_velocity",
        ),
    )
    court = CourtConfig(
        net_post_offset_x=_mapping_float(
            court_raw,
            "net_post_offset_x",
            section="court",
        ),
        net_post_offset_x_range=_optional_number_pair(
            court_raw["net_post_offset_x_range"],
            name="court.net_post_offset_x_range",
        ),
    )
    return GeneratorConfig(
        physics=physics,
        rally=rally,
        camera=camera,
        targeted_velocity=targeted,
        court=court,
    )


@dataclass(frozen=True, slots=True)
class BLCSProposalRejection:
    """One visible rejection inside a bounded proposal sequence."""

    attempt: int
    error_type: str
    reason: str
    from_cell: int
    side: str

    def __post_init__(self) -> None:
        _positive_int(self.attempt, name="attempt")
        _non_empty_text(self.error_type, name="error_type")
        _non_empty_text(self.reason, name="reason")
        _non_negative_int(self.from_cell, name="from_cell")
        if self.side not in {"near", "far"}:
            raise ValueError("side must be 'near' or 'far'.")

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic record."""
        return {
            "attempt": self.attempt,
            "error_type": self.error_type,
            "reason": self.reason,
            "from_cell": self.from_cell,
            "side": self.side,
        }


@dataclass(frozen=True, slots=True)
class BLCSProposalDiagnostic:
    """Bounded proposal outcome for exactly one physical source object."""

    source_trajectory_id: str
    accepted_attempt: int | None
    maximum_attempts: int
    rejected_attempts: tuple[BLCSProposalRejection, ...]

    def __post_init__(self) -> None:
        _identifier(self.source_trajectory_id, name="source_trajectory_id")
        _positive_int(self.maximum_attempts, name="maximum_attempts")
        rejected = tuple(self.rejected_attempts)
        if any(not isinstance(item, BLCSProposalRejection) for item in rejected):
            raise TypeError("rejected_attempts must contain BLCSProposalRejection.")
        if tuple(item.attempt for item in rejected) != tuple(
            range(1, len(rejected) + 1)
        ):
            raise ValueError("Rejected proposal attempts must be contiguous from one.")
        if self.accepted_attempt is None:
            if len(rejected) != self.maximum_attempts:
                raise ValueError(
                    "Exhausted proposals must diagnose every configured attempt."
                )
        else:
            _positive_int(self.accepted_attempt, name="accepted_attempt")
            if self.accepted_attempt > self.maximum_attempts:
                raise ValueError("accepted_attempt must not exceed maximum_attempts.")
            if self.accepted_attempt != len(rejected) + 1:
                raise ValueError(
                    "accepted_attempt must immediately follow rejected attempts."
                )
        object.__setattr__(self, "rejected_attempts", rejected)

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible proposal record."""
        return {
            "source_scene_id": self.source_trajectory_id,
            "accepted_attempt": self.accepted_attempt,
            "maximum_attempts": self.maximum_attempts,
            "rejected_attempts": [
                rejection.to_metadata() for rejection in self.rejected_attempts
            ],
        }


@dataclass(frozen=True, slots=True)
class BLCSPhysicsProvenance:
    """Lossless per-object physics and rally provenance."""

    source_trajectory_id: str
    source_frame_count: int
    initial_from_cell: int
    initial_from_side: str
    rally_length: int
    end_reason: str
    winner_side: str | None
    output_fps: float
    simulation_fps: float
    physics_parameters: Mapping[str, object]
    court_parameters: Mapping[str, object]
    shot_events: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _identifier(self.source_trajectory_id, name="source_trajectory_id")
        _positive_int(self.source_frame_count, name="source_frame_count")
        _non_negative_int(self.initial_from_cell, name="initial_from_cell")
        if self.initial_from_side not in {"near", "far"}:
            raise ValueError("initial_from_side must be 'near' or 'far'.")
        _non_negative_int(self.rally_length, name="rally_length")
        _non_empty_text(self.end_reason, name="end_reason")
        if self.winner_side not in {None, "near", "far"}:
            raise ValueError("winner_side must be None, 'near', or 'far'.")
        output_fps = _positive_float(self.output_fps, name="output_fps")
        simulation_fps = _positive_float(
            self.simulation_fps,
            name="simulation_fps",
        )
        physics = _json_mapping(
            self.physics_parameters,
            name="physics_parameters",
        )
        court = _json_mapping(self.court_parameters, name="court_parameters")
        shots = tuple(
            _json_mapping(shot, name=f"shot_events[{index}]")
            for index, shot in enumerate(self.shot_events)
        )
        object.__setattr__(self, "output_fps", output_fps)
        object.__setattr__(self, "simulation_fps", simulation_fps)
        object.__setattr__(self, "physics_parameters", physics)
        object.__setattr__(self, "court_parameters", court)
        object.__setattr__(self, "shot_events", shots)

    def to_metadata(self) -> dict[str, object]:
        """Return complete JSON-compatible per-object provenance."""
        return {
            "source_trajectory_id": self.source_trajectory_id,
            "source_frame_count": self.source_frame_count,
            "initial_from_cell": self.initial_from_cell,
            "initial_from_side": self.initial_from_side,
            "rally_length": self.rally_length,
            "end_reason": self.end_reason,
            "winner_side": self.winner_side,
            "output_fps": self.output_fps,
            "simulation_fps": self.simulation_fps,
            "physics_parameters": dict(self.physics_parameters),
            "court_parameters": dict(self.court_parameters),
            "shot_events": [dict(shot) for shot in self.shot_events],
        }


@dataclass(frozen=True, slots=True)
class BLCSSourceTrack:
    """Stable identity and source-frame mapping for one ball column."""

    object_id: str
    source_trajectory_id: str
    source_frame_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        _identifier(self.object_id, name="object_id")
        _identifier(self.source_trajectory_id, name="source_trajectory_id")
        mapping = tuple(self.source_frame_indices)
        if not mapping:
            raise ValueError("source_frame_indices must not be empty.")
        active = [value for value in mapping if value is not None]
        if not active:
            raise ValueError("Every BLCS source track must be present.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in active
        ):
            raise TypeError("Source frame indices must be non-negative integers.")
        if active != list(range(active[0], active[0] + len(active))):
            raise ValueError("Source frame indices must be consecutive.")
        active_global = [index for index, value in enumerate(mapping) if value is not None]
        if active_global != list(range(active_global[0], active_global[-1] + 1)):
            raise ValueError("BLCS presence must be one continuous interval.")
        object.__setattr__(self, "source_frame_indices", mapping)

    def to_metadata(self) -> dict[str, object]:
        """Return the complete identity and source-frame mapping."""
        return {
            "object_id": self.object_id,
            "source_trajectory_id": self.source_trajectory_id,
            "source_frame_indices": list(self.source_frame_indices),
        }


@dataclass(frozen=True, slots=True)
class BLCSSourceScene:
    """Complete validated multi-object physics source scene."""

    scene_id: str
    seed: int
    frame_indices: tuple[int, ...]
    fps: float
    simulation_fps: float
    positions_court_m: NDArray[np.float64]
    velocities_court_mps: NDArray[np.float64]
    present: NDArray[np.bool_]
    tracks: tuple[BLCSSourceTrack, ...]
    physics_provenance: tuple[BLCSPhysicsProvenance, ...]
    proposal_diagnostics: tuple[BLCSProposalDiagnostic, ...]

    def __post_init__(self) -> None:
        _identifier(self.scene_id, name="scene_id")
        _non_negative_int(self.seed, name="seed")
        fps = _positive_float(self.fps, name="fps")
        simulation_fps = _positive_float(self.simulation_fps, name="simulation_fps")
        positions = _float64_array(self.positions_court_m, name="positions_court_m")
        velocities = _float64_array(
            self.velocities_court_mps,
            name="velocities_court_mps",
        )
        presence = _bool_array(self.present, name="present")
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions_court_m must have shape [T, O, 3].")
        if velocities.shape != positions.shape:
            raise ValueError("velocities_court_mps must match positions_court_m.")
        if presence.shape != positions.shape[:2]:
            raise ValueError("present must have shape [T, O].")
        if positions.shape[0] <= 0 or positions.shape[1] < 2:
            raise ValueError(
                "BLCS source scenes require at least one frame and two objects."
            )
        frame_indices = tuple(self.frame_indices)
        if frame_indices != tuple(range(positions.shape[0])):
            raise ValueError("frame_indices must cover every frame exactly as 0..T-1.")
        tracks = tuple(self.tracks)
        provenance = tuple(self.physics_provenance)
        diagnostics = tuple(self.proposal_diagnostics)
        object_count = positions.shape[1]
        if not (
            len(tracks)
            == len(provenance)
            == len(diagnostics)
            == object_count
        ):
            raise ValueError(
                "Tracks, provenance, and diagnostics must match the object axis."
            )
        if any(not isinstance(item, BLCSSourceTrack) for item in tracks):
            raise TypeError("tracks must contain BLCSSourceTrack values.")
        if any(not isinstance(item, BLCSPhysicsProvenance) for item in provenance):
            raise TypeError(
                "physics_provenance must contain BLCSPhysicsProvenance values."
            )
        if any(not isinstance(item, BLCSProposalDiagnostic) for item in diagnostics):
            raise TypeError(
                "proposal_diagnostics must contain BLCSProposalDiagnostic values."
            )
        object_ids = [track.object_id for track in tracks]
        if len(set(object_ids)) != object_count:
            raise ValueError("BLCS source object identities must be unique.")
        source_ids = [track.source_trajectory_id for track in tracks]
        if source_ids != [item.source_trajectory_id for item in provenance]:
            raise ValueError("Track identity and physics provenance order disagree.")
        if source_ids != [item.source_trajectory_id for item in diagnostics]:
            raise ValueError("Track identity and proposal diagnostic order disagree.")
        for object_index, track in enumerate(tracks):
            if len(track.source_frame_indices) != positions.shape[0]:
                raise ValueError("Every source-frame mapping must cover all frames.")
            mapped_presence = np.asarray(
                [value is not None for value in track.source_frame_indices],
                dtype=np.bool_,
            )
            if not np.array_equal(mapped_presence, presence[:, object_index]):
                raise ValueError("Presence disagrees with a source-frame mapping.")
            active = [
                value for value in track.source_frame_indices if value is not None
            ]
            if active[-1] >= provenance[object_index].source_frame_count:
                raise ValueError("A source-frame mapping exceeds its physics trajectory.")
            if provenance[object_index].output_fps != fps:
                raise ValueError("All source objects must share the scene output fps.")
            if provenance[object_index].simulation_fps != simulation_fps:
                raise ValueError("All source objects must share the simulation fps.")
            if diagnostics[object_index].accepted_attempt is None:
                raise ValueError("Published source scenes cannot contain exhausted proposals.")
        for array in (positions, velocities, presence):
            array.setflags(write=False)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "simulation_fps", simulation_fps)
        object.__setattr__(self, "positions_court_m", positions)
        object.__setattr__(self, "velocities_court_mps", velocities)
        object.__setattr__(self, "present", presence)
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "physics_provenance", provenance)
        object.__setattr__(self, "proposal_diagnostics", diagnostics)

    @property
    def frame_count(self) -> int:
        """Return the complete global source timeline length."""
        return int(self.positions_court_m.shape[0])

    @property
    def object_count(self) -> int:
        """Return the number of stable physical ball identities."""
        return int(self.positions_court_m.shape[1])

    def to_metadata(self) -> dict[str, object]:
        """Return source provenance without embedding trajectory arrays."""
        return {
            "generator": "blcs_physics",
            "source_scene": self.scene_id,
            "seed": self.seed,
            "source_frame_count": self.frame_count,
            "fps": self.fps,
            "simulation_fps": self.simulation_fps,
            "tracks": [track.to_metadata() for track in self.tracks],
            "physics_sources": [
                provenance.to_metadata() for provenance in self.physics_provenance
            ],
            "physics_proposals": [
                diagnostic.to_metadata() for diagnostic in self.proposal_diagnostics
            ],
        }


@dataclass(slots=True)
class _BoundedPhysicsSceneSource:
    source: BLCSSceneGenerator
    maximum_attempts: int
    diagnostics: dict[str, BLCSProposalDiagnostic] = field(default_factory=dict)
    provenance: dict[str, BLCSPhysicsProvenance] = field(default_factory=dict)
    config: GeneratorConfig = field(init=False)

    def __post_init__(self) -> None:
        self.config = self.source.config

    def sample_from_cell(self) -> int:
        value = self.source.sample_from_cell()
        return _non_negative_int(value, name="sampled from_cell")

    def sample_side(self) -> str:
        value = self.source.sample_side()
        if value not in {"near", "far"}:
            raise ValueError("Sampled BLCS side must be 'near' or 'far'.")
        return str(value)

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData | None:
        rejected: list[BLCSProposalRejection] = []
        for attempt in range(1, self.maximum_attempts + 1):
            if attempt > 1:
                from_cell = self.sample_from_cell()
                side = self.sample_side()
            try:
                scene = self.source.generate_scene(from_cell, side, scene_id)
            except BLCSPhysicsProposalRejected as error:
                rejected.append(
                    _proposal_rejection(
                        attempt=attempt,
                        error=error,
                        from_cell=from_cell,
                        side=side,
                    )
                )
                continue
            except RuntimeError as error:
                if not is_retryable_full_physics_rejection(error):
                    raise
                rejected.append(
                    _proposal_rejection(
                        attempt=attempt,
                        error=error,
                        from_cell=from_cell,
                        side=side,
                    )
                )
                continue
            if scene is None:
                rejected.append(
                    BLCSProposalRejection(
                        attempt=attempt,
                        error_type="EmptyPhysicsProposal",
                        reason="BLCS physical scene generation returned no scene.",
                        from_cell=from_cell,
                        side=side,
                    )
                )
                continue
            if scene.scene_id != scene_id:
                raise ValueError("BLCS physics generator changed the requested scene ID.")
            diagnostic = BLCSProposalDiagnostic(
                source_trajectory_id=scene_id,
                accepted_attempt=attempt,
                maximum_attempts=self.maximum_attempts,
                rejected_attempts=tuple(rejected),
            )
            if scene_id in self.diagnostics or scene_id in self.provenance:
                raise ValueError("BLCS physics generator reused a source trajectory ID.")
            self.diagnostics[scene_id] = diagnostic
            self.provenance[scene_id] = _physics_provenance(scene)
            return scene
        diagnostic = BLCSProposalDiagnostic(
            source_trajectory_id=scene_id,
            accepted_attempt=None,
            maximum_attempts=self.maximum_attempts,
            rejected_attempts=tuple(rejected),
        )
        raise BLCSPhysicsProposalExhausted(diagnostic)


@dataclass(frozen=True, slots=True)
class BLCSPhysicsTrajectorySource:
    """Adapt task-local generators to the stable complete-source contract."""

    generator_config: BLCSGeneratorConfiguration
    settings: BLCSPhysicsSourceSettings

    def __post_init__(self) -> None:
        if not isinstance(self.generator_config, GeneratorConfig):
            raise TypeError("generator_config must be a BLCS generator configuration.")
        if not isinstance(self.settings, BLCSPhysicsSourceSettings):
            raise TypeError("settings must be BLCSPhysicsSourceSettings.")

    def preflight(self, *, scene_id: str, seed: int) -> None:
        """Validate a source request without advancing any random generator."""
        _identifier(scene_id, name="scene_id")
        _non_negative_int(seed, name="seed")

    def generate(self, *, scene_id: str, seed: int) -> BLCSSourceScene:
        """Generate one deterministic complete multi-object source scene."""
        self.preflight(scene_id=scene_id, seed=seed)
        with _deterministic_random_state(seed):
            bounded = _BoundedPhysicsSceneSource(
                source=BLCSSceneGenerator(
                    config=self.generator_config,
                    device=self.settings.device,
                ),
                maximum_attempts=(
                    self.settings.maximum_physics_attempts_per_object
                ),
            )
            internal = MultiBallSceneGenerator(
                bounded,
                timeline=self.settings.timeline._to_internal(),
                # ``bounded`` already owns the configured proposal sequence and
                # its lossless diagnostics. The compositor must not add a
                # second, unrecorded retry layer around that source contract.
                maximum_physics_attempts_per_object=1,
                rng=random.Random(seed),
            ).generate_scene(scene_id)
        return _source_scene_from_internal(
            internal,
            seed=seed,
            diagnostics=bounded.diagnostics,
            provenance=bounded.provenance,
        )


def _source_scene_from_internal(
    scene: BLCSSceneData,
    *,
    seed: int,
    diagnostics: Mapping[str, BLCSProposalDiagnostic],
    provenance: Mapping[str, BLCSPhysicsProvenance],
) -> BLCSSourceScene:
    raw_positions = _source_array(scene.ball_pos_world, name="ball_pos_world")
    raw_velocities = _source_array(scene.ball_vel_world, name="ball_vel_world")
    raw_presence = _source_array(scene.ball_present, name="ball_present")
    if (
        isinstance(scene.num_balls, bool)
        or not isinstance(scene.num_balls, int)
        or scene.num_balls < 2
    ):
        raise ValueError("BLCS source scene did not contain multiple physical balls.")
    if raw_positions.ndim != 3 or raw_positions.shape[1] < scene.num_balls:
        raise ValueError("BLCS source positions do not cover every physical ball.")
    if raw_velocities.shape != raw_positions.shape:
        raise ValueError("BLCS source velocities do not match positions.")
    if (
        raw_presence.dtype != np.bool_
        or raw_presence.shape != raw_positions.shape[:2]
    ):
        raise ValueError("BLCS source presence does not match trajectory shape.")
    positions = _float64_array(
        raw_positions[:, : scene.num_balls],
        name="ball_pos_world",
    )
    velocities = _float64_array(
        raw_velocities[:, : scene.num_balls],
        name="ball_vel_world",
    )
    presence = _bool_array(
        raw_presence[:, : scene.num_balls],
        name="ball_present",
    )
    tracks = _tracks_from_internal_scene(
        frame_count=int(positions.shape[0]),
        object_count=scene.num_balls,
        presence=presence,
        placements=scene.track_instances,
    )
    source_ids = tuple(track.source_trajectory_id for track in tracks)
    if set(diagnostics) != set(source_ids):
        raise ValueError("BLCS proposal diagnostics do not match composed identities.")
    if set(provenance) != set(source_ids):
        raise ValueError("BLCS physics provenance does not match composed identities.")
    return BLCSSourceScene(
        scene_id=scene.scene_id,
        seed=seed,
        frame_indices=tuple(range(positions.shape[0])),
        fps=float(scene.fps_out),
        simulation_fps=float(scene.sim_fps),
        positions_court_m=positions,
        velocities_court_mps=velocities,
        present=presence,
        tracks=tracks,
        physics_provenance=tuple(provenance[source_id] for source_id in source_ids),
        proposal_diagnostics=tuple(
            diagnostics[source_id] for source_id in source_ids
        ),
    )


def _tracks_from_internal_scene(
    *,
    frame_count: int,
    object_count: int,
    presence: NDArray[np.bool_],
    placements: Sequence[Mapping[str, object]],
) -> tuple[BLCSSourceTrack, ...]:
    if len(placements) != object_count:
        raise ValueError("track_instances must contain one record per ball column.")
    required = {
        "track_id",
        "source_scene_id",
        "source_start",
        "source_end",
        "birth_frame",
        "death_frame",
    }
    by_track: dict[int, Mapping[str, object]] = {}
    for placement in placements:
        if set(placement) != required:
            raise ValueError("track_instances contains unknown or missing fields.")
        track_id = placement["track_id"]
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise TypeError("track_id must be an integer.")
        if track_id in by_track:
            raise ValueError("track_instances contains duplicate track_id values.")
        by_track[track_id] = placement
    if set(by_track) != set(range(object_count)):
        raise ValueError("track_id values must equal the physical object columns.")
    tracks: list[BLCSSourceTrack] = []
    for track_id in range(object_count):
        placement = by_track[track_id]
        source_id = placement["source_scene_id"]
        if not isinstance(source_id, str):
            raise TypeError("source_scene_id must be a string.")
        source_start = _non_negative_int(
            placement["source_start"],
            name="source_start",
        )
        source_end = _non_negative_int(
            placement["source_end"],
            name="source_end",
        )
        birth = _non_negative_int(placement["birth_frame"], name="birth_frame")
        death = _non_negative_int(placement["death_frame"], name="death_frame")
        if not (0 <= birth < death <= frame_count):
            raise ValueError("BLCS track interval is outside the global timeline.")
        if source_end - source_start != death - birth:
            raise ValueError("BLCS source and global intervals differ in length.")
        expected_presence: NDArray[np.bool_] = np.zeros(
            frame_count,
            dtype=np.bool_,
        )
        expected_presence[birth:death] = True
        if not np.array_equal(expected_presence, presence[:, track_id]):
            raise ValueError("track_instances disagrees with ball_present.")
        mapping: list[int | None] = [None] * frame_count
        mapping[birth:death] = range(source_start, source_end)
        tracks.append(
            BLCSSourceTrack(
                object_id=f"ball-{track_id + 1:03d}",
                source_trajectory_id=source_id,
                source_frame_indices=tuple(mapping),
            )
        )
    return tuple(tracks)


def _physics_provenance(scene: BLCSSceneData) -> BLCSPhysicsProvenance:
    positions = _source_array(scene.ball_pos_world, name="ball_pos_world")
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("Physical source proposals must have shape [T, 3].")
    return BLCSPhysicsProvenance(
        source_trajectory_id=scene.scene_id,
        source_frame_count=int(positions.shape[0]),
        initial_from_cell=scene.initial_from_cell,
        initial_from_side=scene.initial_from_side,
        rally_length=scene.rally_length,
        end_reason=scene.end_reason,
        winner_side=scene.winner_side,
        output_fps=float(scene.fps_out),
        simulation_fps=float(scene.sim_fps),
        physics_parameters=scene.physics_config_dict,
        court_parameters=scene.court_config_dict,
        shot_events=tuple(scene.shots),
    )


def _proposal_rejection(
    *,
    attempt: int,
    error: RuntimeError,
    from_cell: int,
    side: str,
) -> BLCSProposalRejection:
    reason = str(error).strip()
    return BLCSProposalRejection(
        attempt=attempt,
        error_type=type(error).__name__,
        reason=reason if reason else "Physics proposal rejected without a reason.",
        from_cell=from_cell,
        side=side,
    )


@contextmanager
def _deterministic_random_state(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _source_array(value: object, *, name: str) -> NDArray[Any]:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    if value is None:
        raise TypeError(f"{name} must not be None.")
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise TypeError(f"{name} must be a numeric or boolean array.")
    return array


def _float64_array(value: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _bool_array(value: object, *, name: str) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must use bool dtype.")
    return np.array(array, dtype=np.bool_, order="C", copy=True)


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys do not match; "
            f"missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}."
        )
    return value


def _mapping_int(value: Mapping[str, object], key: str) -> int:
    return _integer(value[key], name=f"timeline.{key}")


def _mapping_section(
    value: Mapping[str, object],
    key: str,
    *,
    keys: set[str],
) -> Mapping[str, object]:
    return _exact_mapping(
        value[key],
        keys=keys,
        name=f"BLCS generator configuration.{key}",
    )


def _mapping_text(value: Mapping[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip() or item != item.strip():
        raise TypeError(f"{key} must be a non-empty trimmed string.")
    return item


def _mapping_bool(
    value: Mapping[str, object],
    key: str,
    *,
    section: str,
) -> bool:
    item = value[key]
    if not isinstance(item, bool):
        raise TypeError(f"{section}.{key} must be a boolean.")
    return item


def _mapping_float(
    value: Mapping[str, object],
    key: str,
    *,
    section: str,
) -> float:
    return _finite_float(value[key], name=f"{section}.{key}")


def _positive_mapping_float(
    value: Mapping[str, object],
    key: str,
    *,
    section: str,
) -> float:
    return _positive_float(value[key], name=f"{section}.{key}")


def _non_negative_mapping_float(
    value: Mapping[str, object],
    key: str,
    *,
    section: str,
) -> float:
    result = _finite_float(value[key], name=f"{section}.{key}")
    if result < 0.0:
        raise ValueError(f"{section}.{key} must be non-negative.")
    return result


def _positive_mapping_int(
    value: Mapping[str, object],
    key: str,
    *,
    section: str,
) -> int:
    return _positive_int(value[key], name=f"{section}.{key}")


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _non_negative_int(value: object, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _integer_pair(value: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{name} must contain exactly two integers.")
    return (
        _integer(value[0], name=f"{name}[0]"),
        _integer(value[1], name=f"{name}[1]"),
    )


def _positive_integer_pair(value: object, *, name: str) -> tuple[int, int]:
    pair = _integer_pair(value, name=name)
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be positive.")
    return pair


def _number_pair(value: object, *, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{name} must contain exactly two numbers.")
    pair = (
        _finite_float(value[0], name=f"{name}[0]"),
        _finite_float(value[1], name=f"{name}[1]"),
    )
    if pair[0] > pair[1]:
        raise ValueError(f"{name} must be ordered.")
    return pair


def _number_triple(value: object, *, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise TypeError(f"{name} must contain exactly three numbers.")
    return (
        _finite_float(value[0], name=f"{name}[0]"),
        _finite_float(value[1], name=f"{name}[1]"),
        _finite_float(value[2], name=f"{name}[2]"),
    )


def _optional_number_pair(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    pair = _number_pair(value, name=name)
    if minimum is not None and pair[0] < minimum:
        raise ValueError(f"{name} values must be at least {minimum}.")
    return pair


def _probability(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return result


def _field_of_view(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 < result < 180.0:
        raise ValueError(f"{name} must be inside (0, 180) degrees.")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _PORTABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable non-empty identifier.")
    return value


def _non_empty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text.")
    return value


def _json_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return {
        key: _json_value(item, name=f"{name}.{key}") for key, item in value.items()
    }


def _json_value(value: object, *, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} contains a non-finite number.")
        return result
    if isinstance(value, Mapping):
        return _json_mapping(value, name=name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item, name=name) for item in value]
    raise TypeError(f"{name} must be JSON-compatible, got {type(value).__name__}.")


__all__ = [
    "BLCSGeneratorConfiguration",
    "BLCSPhysicsProposalExhausted",
    "BLCSPhysicsProposalRejected",
    "BLCSPhysicsProvenance",
    "BLCSPhysicsSourceSettings",
    "BLCSPhysicsTrajectorySource",
    "BLCSProposalDiagnostic",
    "BLCSProposalRejection",
    "BLCSSourceScene",
    "BLCSSourceTrack",
    "BLCSTimelineSpec",
    "build_blcs_generator_configuration",
]
