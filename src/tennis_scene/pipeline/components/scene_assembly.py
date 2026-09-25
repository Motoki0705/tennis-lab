"""Final typed SceneResult assembly from committed component results."""

from __future__ import annotations

from dataclasses import dataclass

from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationOutput,
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.tennis_scene.schema import SceneResult
from src.utils.video import VideoInfo


@dataclass(frozen=True)
class SceneAssemblyInput:
    source: ClipSource
    calibration: CourtCalibrationOutput
    alignment: CameraAlignmentOutput
    people: GroupedObservations
    skeleton: PlayerTriangulationOutput
    ball: BallTriangulationOutput
    placement: BodyPlacementOutput


class SceneAssemblyModule:
    def __init__(self, camera_ids: tuple[str, ...], *, require_people: bool, require_ball: bool, require_body: bool) -> None:
        self.require_people, self.require_ball, self.require_body = require_people, require_ball, require_body
        self.io = ComponentIO("scene_assembly", SceneAssemblyInput, SceneResult,
            {"calibration": InputPort("local_court_calibration"), "alignment": InputPort("aligned_cameras"),
             "identities": InputPort("person_identities"), "skeleton": InputPort("player_skeletons"),
             "ball": InputPort("ball_trajectory"), "placement": InputPort("placed_bodies"),
             **{f"pose_{c}": InputPort("person_poses") for c in camera_ids}}, "scene_result", 2)

    def process(self, inputs: SceneAssemblyInput) -> SceneResult:
        source = inputs.source
        geometry, skeleton, ball, players = inputs.alignment.geometry, inputs.skeleton.skeleton, inputs.ball.ball, inputs.placement.players
        active = tuple(v.source_index for v in inputs.calibration.calibration.views)
        if geometry is None:
            status = "empty"
        else:
            has_people = skeleton is not None and bool(skeleton.valid.any())
            has_ball = ball is not None and bool(ball.trajectory.valid.any())
            has_body = players is not None and bool(players.smpl_valid.any())
            if not has_people and not has_ball:
                raise ReconstructionUnavailable("no_valid_reconstruction", "Nonempty observations produced no supported 3D geometry")
            complete = (not self.require_people or has_people) and (not self.require_ball or has_ball) and (not self.require_body or has_body)
            status = "ok" if complete else "partial"
        first = source.videos[0]
        return assemble_automatic_scene(video_paths=source.paths, camera_ids=source.camera_ids,
            info=VideoInfo(first.fps, first.width, first.height, first.num_frames), court=inputs.calibration.court,
            active_indices=active, geometry=geometry, grouped_people=inputs.people,
            skeleton=skeleton, players=players, ball=ball,
            metadata={"status": status, "dataset_clip_id": source.clip_id, "pipeline_contract": "declared_components_v1"})
