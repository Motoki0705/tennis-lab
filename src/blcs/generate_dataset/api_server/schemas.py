"""Pydantic schemas for the BLCS simulation WebUI API.

We use JSON-friendly types (lists and dicts) to keep the API easy to call
from Next.js without custom binary encoders.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Side = Literal["near", "far"]
TargetMode = Literal["none", "cell", "point"]


class Vec3(BaseModel):
    """3D vector in meters (pos) or SI units (vel/spin)."""

    x: float
    y: float
    z: float


class Vec2(BaseModel):
    """2D vector in meters (x,y) on the court plane."""

    x: float
    y: float


class PhysicsParams(BaseModel):
    gravity: float | None = None
    k_drag: float | None = None
    k_magnus: float | None = None
    e_z: float | None = None
    mu: float | None = None
    alpha_net: float | None = None
    dt: float | None = None
    use_drag: bool | None = None
    use_magnus: bool | None = None


class SimParams(BaseModel):
    max_sim_frames: int | None = None
    sim_fps: int | None = None
    output_fps: int | None = None


class ShotParams(BaseModel):
    """Parameters for initial state overrides.

    - If a field is omitted, server picks a default/sample.
    - If velocity is not provided and target_mode is `cell`/`point`, the server
      will compute a targeted velocity (current implementation).
    """

    position: Vec3 | None = None
    velocity: Vec3 | None = None
    spin: Vec3 | None = None


class SimulateShotRequest(BaseModel):
    from_side: Side
    from_cell: int = Field(ge=0, le=19)

    target_mode: TargetMode = "none"
    to_cell: int | None = Field(default=None, ge=0, le=19)
    target_point: Vec2 | None = None

    shot: ShotParams = Field(default_factory=ShotParams)
    physics: PhysicsParams = Field(default_factory=PhysicsParams)
    sim: SimParams = Field(default_factory=SimParams)

    seed: int | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> SimulateShotRequest:
        if self.target_mode == "cell" and self.to_cell is None:
            raise ValueError("to_cell is required when target_mode='cell'")
        if self.target_mode == "point" and self.target_point is None:
            raise ValueError("target_point is required when target_mode='point'")
        return self


class ShotEvents(BaseModel):
    t_net: int
    t_fence: int
    t_bounce1: int
    t_bounce2: int

    net_pos: Vec3 | None
    bounce1_pos: Vec3 | None
    bounce2_pos: Vec3 | None


class ShotLabels(BaseModel):
    category: str
    to_cell: int | None


class ShotMetrics(BaseModel):
    apex_height_m: float
    time_to_bounce1_s: float | None
    net_clearance_m: float | None


class SimulateShotResponse(BaseModel):
    positions: list[list[float]]  # [T][3]
    velocities: list[list[float]]  # [T][3]
    fps_out: int
    sim_fps: int

    events: ShotEvents
    labels: ShotLabels
    metrics: ShotMetrics


class CellBounds(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class CellInfo(BaseModel):
    cell_id: int = Field(ge=0, le=19)
    side: Side
    bounds: CellBounds
    center: Vec2


class CellsResponse(BaseModel):
    cells: list[CellInfo]


class CourtGeometryResponse(BaseModel):
    """3D geometry hints for rendering a tennis court in the WebUI.

    - keypoints are CourtKP20 points from `src.utils.geometry.court.court_keypoints_3d`.
    - segments are pairs of indices into `keypoints` to draw as line segments.
    """

    keypoints: list[list[float]]  # [20][3]
    segments: list[list[int]]  # [[i, j], ...]
