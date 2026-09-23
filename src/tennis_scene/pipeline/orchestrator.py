"""Unattended camera-local association, triangulation and calibrated body recovery."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.submodules.models import SmplCoco17Reconstructor
from src.tasks.base.model_io.association_contracts import (
    AssociationObservationRequest,
    AssociationObservationResult,
)
from src.tennis_scene.pipeline.artifacts import (
    PipelineArtifactStore,
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionResult,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
    reconstruct_rally_ball,
)
from src.tennis_scene.pipeline.components.camera_geometry import (
    SideEvidence,
    calibrate_local_courts,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule, CourtKPResult
from src.tennis_scene.pipeline.components.person_observations import (
    PersonObservationModule,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    BodyRecovery,
    PlayerSkeleton,
    ReconstructedPlayers,
    reconstruct_player_bodies,
    triangulate_players,
)
from src.tennis_scene.pipeline.components.view_association import ViewAssociationModule
from src.tennis_scene.pipeline.dependency_graph import (
    build_default_dependency_graph,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.model_io.body import BodyRecoveryAdapter
from src.tennis_scene.pipeline.model_io.observations import (
    GroupedObservations,
    ObjectObservations,
    group_observations,
)
from src.tennis_scene.pipeline.model_io.people import (
    PersonObservationAdapter,
    build_people_chain,
)
from src.tennis_scene.pipeline.utilts.timeline import (
    association_frame_indices,
    restore_source_ids,
)
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.geometry.triangulation import TriangulatedPoints
from src.utils.paths import PROJECT_ROOT
from src.utils.video import VideoInfo, probe_video_info

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tennis_scene.configuration import PipelineRuntimeConfig

LOGGER = logging.getLogger(__name__)


class TennisSceneOrchestrator:
    def __init__(
        self, config: PipelineRuntimeConfig, *,
        court: CourtKPModule,
        people: PersonObservationModule | None,
        ball: BallDetectionModule | None,
        plcs: ViewAssociationModule | None,
        blcs: ViewAssociationModule | None,
        body: BodyRecovery | None,
    ) -> None:
        self.config = config
        self.court, self.people, self.ball = court, people, ball
        self.plcs, self.blcs, self.body = plcs, blcs, body
        self.resolution = build_default_dependency_graph(config.enabled).resolve_from_enabled(config.enabled)
        self.enabled_stages = self.resolution.enabled_set
        self.execution_order = self.resolution.enabled_order
        self.last_receipt: dict[str, Any] = {}
        self._file_identities: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}
        source = PROJECT_ROOT / "src"
        self.code_identity = document_digest({str(path.relative_to(source)): dual_sha256(path) for path in sorted(source.rglob("*.py"))})

    @classmethod
    def from_runtime_config(cls, cfg: PipelineRuntimeConfig) -> TennisSceneOrchestrator:
        chain = build_people_chain(cfg.people) if cfg.enabled["person_observations"] or cfg.enabled["gvhmr"] else None
        people = None if chain is None else PersonObservationModule(PersonObservationAdapter(chain, bbox_enlarge=cfg.people.runtime.tracking.bbox_enlarge))
        body = None
        if chain is not None and cfg.enabled["gvhmr"]:
            coco = SmplCoco17Reconstructor(cfg.people.body_models_dir, device=cfg.device, bundled_assets=cfg.people.bundled_assets)
            body = BodyRecoveryAdapter(chain, coco, cfg.people.bundled_assets.smpl_neutral_joint_regressor)
        return cls(
            cfg, court=CourtKPModule(cfg.court_kp),
            people=people if cfg.enabled["person_observations"] else None,
            ball=BallDetectionModule(cfg.ball_detection) if cfg.enabled["ball_detection"] else None,
            plcs=ViewAssociationModule(cfg.plcs_checkpoint, task="plcs", device=cfg.device) if cfg.enabled["plcs_association"] else None,
            blcs=ViewAssociationModule(cfg.blcs_checkpoint, task="blcs", device=cfg.device) if cfg.enabled["blcs_association"] else None,
            body=body,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TennisSceneOrchestrator:
        from src.tennis_scene.configuration import PipelineRuntimeConfig
        return cls.from_runtime_config(PipelineRuntimeConfig.from_config(cfg))

    def _file_identity(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"path": str(path), "sha256": None, "state": "absent"}
        stat = path.stat()
        version = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        cached = self._file_identities.get(path)
        if cached is None or cached[0] != version:
            self._file_identities[path] = version, dual_sha256(path)
        return {"path": str(path), "sha256": self._file_identities[path][1]}

    def publication_identity(self) -> dict[str, Any]:
        """Functional settings and asset bytes, independent of output directories."""
        cfg = self.config
        checkpoints = {
            "court": cfg.court_kp.checkpoint, "dino": cfg.people.dino_checkpoint,
            "vitpose": cfg.people.vitpose_checkpoint, "hmr2": cfg.people.hmr2_checkpoint,
            "gvhmr": cfg.people.gvhmr_checkpoint, "plcs_association": cfg.plcs_checkpoint,
            "blcs_association": cfg.blcs_checkpoint, "ball_detection": cfg.ball_detection.checkpoint,
            "smplx": cfg.people.body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz",
            "root_regressor": cfg.people.bundled_assets.smpl_neutral_joint_regressor,
        }
        return {"schema": "automatic_pipeline_v2", "code_sha256": self.code_identity,
                "settings": json_value(cfg.processing_settings),
                "checkpoints": {name: self._file_identity(path) for name, path in checkpoints.items()}}

    @staticmethod
    def _empty_objects(ids: tuple[str, ...], info: VideoInfo, frames: int, joints: int) -> ObjectObservations:
        shape = (len(ids), frames, 0, joints)
        return ObjectObservations(ids, (info.width, info.height), info.fps, np.zeros((*shape, 2), np.float32), np.zeros(shape, np.float32), np.zeros(shape[:3], bool), np.empty((len(ids), 0), np.int64))

    def _probe_synced_video_infos(self, video_paths: Sequence[Path], *, max_frames: int | None) -> list[VideoInfo]:
        infos = [probe_video_info(path) for path in video_paths]
        if not infos:
            raise ValueError("No source videos")
        first = infos[0]
        expected = first.frame_count if max_frames is None else min(first.frame_count, max_frames)
        for info in infos:
            frames = info.frame_count if max_frames is None else min(info.frame_count, max_frames)
            if frames != expected or abs(info.fps - first.fps) > 1e-6 or (info.width, info.height) != (first.width, first.height):
                raise ValueError("Videos must have synchronized frame counts, FPS and image size")
        if expected < 1 or first.fps < 28:
            raise ValueError("Automatic reconstruction requires nonempty synchronized videos at >=28fps")
        return infos

    def _associate(
        self, module: ViewAssociationModule | None, task: str, raw: ObjectObservations,
        court: CourtKPResult, active_indices: tuple[int, ...], sample_frames: np.ndarray,
        reference: str, store: PipelineArtifactStore, common: dict[str, Any], statuses: dict[str, str],
    ) -> tuple[AssociationObservationResult | None, GroupedObservations]:
        threshold = self.config.human_vis_threshold if task == "plcs" else self.config.ball_detection.score_threshold
        uv, visible = raw.normalized(threshold)
        stage = f"{task}_association"
        self.last_receipt["active_stage"] = stage
        if module is None or not visible.any():
            statuses[stage] = "disabled" if module is None else "skipped_no_observations"
            return None, group_observations(raw, np.full(raw.observed.shape, -1, np.int64), threshold=threshold)
        identity = {
            **common, "stage": stage, "checkpoint": self._file_identity(module.checkpoint),
            "observations": store.references.get("people" if task == "plcs" else "ball"),
            "court": store.references["court"], "policy": json_value(self.config.inference_policy),
            "camera_ids": list(raw.camera_ids), "reference": reference, "source_frames": sample_frames.tolist(),
            "visibility_threshold": threshold,
        }
        cached = store.load(stage, identity)
        if cached is None:
            size = np.asarray(raw.size, np.float32)
            court_uv = court.keypoints[list(active_indices)][:, sample_frames] * (np.maximum(size - 1, 1) / size)
            request = AssociationObservationRequest(
                torch.from_numpy(uv[:, sample_frames]), torch.from_numpy(visible[:, sample_frames]),
                torch.from_numpy(court_uv.astype(np.float32)),
                torch.from_numpy(court.visibility[list(active_indices)][:, sample_frames].astype(bool)),
                raw.camera_ids, reference, torch.from_numpy(sample_frames),
            )
            result = module.process_observations(request, policy=self.config.inference_policy)
            if result is None or module.tracking is None:
                raise RuntimeError("Nonempty association input returned no inference result")
            tracking = module.tracking
            result_fields = {field.name: (getattr(result, field.name).numpy() if isinstance(getattr(result, field.name), torch.Tensor) else getattr(result, field.name)) for field in fields(result)}
            dense_ids = restore_source_ids(uv, visible, result.raw_ids.numpy(), sample_frames, tracking=tracking)
            store.save(stage, identity, {"result": result_fields, "tracking": asdict(tracking), "source_ids": dense_ids})
            statuses[stage] = "executed"
        else:
            values = cached["result"]
            restored: dict[str, Any] = {name: (torch.from_numpy(value) if isinstance(value, np.ndarray) else tuple(value) if name == "camera_ids" else value) for name, value in values.items()}
            result = AssociationObservationResult(**restored)
            dense_ids = np.asarray(cached["source_ids"], np.int64)
            statuses[stage] = "loaded"
        return result, group_observations(raw, dense_ids, threshold=threshold)

    def run(
        self, video_paths: Sequence[Path], *, video_role: PathRole, camera_ids: Sequence[str],
        max_frames: int | None = None,
    ) -> SceneResult:
        self.last_receipt = {}
        try:
            return self._run(video_paths, video_role=video_role, camera_ids=camera_ids, max_frames=max_frames)
        except Exception as exc:
            if not self.last_receipt:
                reason = exc.reason if isinstance(exc, ReconstructionUnavailable) else "input_contract_error"
                self.last_receipt = {"schema": "tennis_scene_run_v2", "status": "failed", "reason": reason,
                    "error": str(exc), "video_paths": [str(path) for path in video_paths], "camera_ids": list(camera_ids)}
                key = document_digest({"paths": [str(path) for path in video_paths], "ids": list(camera_ids), "max_frames": max_frames})[:20]
                write_json_atomic(self.config.cache_directory / "preflight" / key / "run.json", self.last_receipt)
            raise

    def _run(
        self, video_paths: Sequence[Path], *, video_role: PathRole, camera_ids: Sequence[str],
        max_frames: int | None = None,
    ) -> SceneResult:
        cfg = self.config
        paths = tuple(cfg.resolver.validate(video_role, Path(path)) for path in video_paths)
        ids = tuple(camera_ids)
        if not 3 <= len(paths) <= 5 or len(ids) != len(paths) or len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError("Automatic reconstruction requires 3..5 unique camera IDs and videos")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        if cfg.camera_geometry.reference_camera is not None and cfg.camera_geometry.reference_camera not in ids:
            raise ValueError("Reference camera is not in the source camera IDs")
        infos = self._probe_synced_video_infos(paths, max_frames=max_frames)
        info = infos[0]
        frames = info.frame_count if max_frames is None else min(info.frame_count, max_frames)
        sample_frames = association_frame_indices(frames, info.fps, max_frames=cfg.inference_policy.max_frames)
        source = {"videos": [self._file_identity(path) for path in paths], "camera_ids": list(ids), "frames": frames, "fps": info.fps, "size": [info.width, info.height]}
        store = PipelineArtifactStore(cfg.cache_directory / document_digest(source)[:20], source=cfg.cache_source, overwrite=cfg.cache_overwrite)
        common = {"source": source, "code_sha256": self.code_identity, "numpy": np.__version__, "torch": str(torch.__version__)}
        statuses = {key: "pending" if enabled else "disabled" for key, enabled in cfg.enabled.items()}
        timings: dict[str, float] = {}
        self.last_receipt = {"schema": "tennis_scene_run_v2", "status": "running", "source": source, "stage_status": statuses, "stage_seconds": timings, "code_sha256": self.code_identity}
        started = time.monotonic()
        try:
            tick = time.monotonic()
            self.last_receipt["active_stage"] = "court_kp"
            court_identity = {**common, "checkpoint": self._file_identity(cfg.court_kp.checkpoint), "settings": cfg.processing_settings["court_kp"]}
            cached = store.load("court", court_identity)
            if cached is None:
                try:
                    court = self.court.process(paths, max_frames=max_frames, annotation_frame_index=0)
                finally:
                    self.court.unload()
                store.save("court", court_identity, {"keypoints": court.keypoints, "visibility": court.visibility, "frame_indices": court.frame_indices, "diagnostics": court.diagnostics})
                statuses["court_kp"] = "executed"
            else:
                court = CourtKPResult.from_dict(cached)
                statuses["court_kp"] = "loaded"
            if court.keypoints.shape != (len(ids), frames, 14, 2) or not np.array_equal(court.frame_indices, np.arange(frames)):
                raise ValueError("Court observations must preserve the complete source timeline")
            calibration = calibrate_local_courts(court, ids, size=(info.width, info.height), config=cfg.camera_geometry)
            timings["court_kp"] = time.monotonic() - tick

            tick = time.monotonic()
            self.last_receipt["active_stage"] = "person_observations"
            people = self._empty_objects(ids, info, frames, 17)
            if self.people is not None:
                people_identity = {**common, "court": store.references["court"], "settings": cfg.processing_settings["person_observations"], "runtime": json_value(cfg.people.runtime),
                    "detector": self._file_identity(cfg.people.dino_checkpoint), "pose": self._file_identity(cfg.people.vitpose_checkpoint)}
                cached = store.load("people", people_identity)
                if cached is None:
                    polygons: list[tuple[tuple[float, float], ...] | None] = [None] * len(ids)
                    for local in calibration.views:
                        polygons[local.source_index] = local.footpoint_polygon(sideline_margin_m=cfg.person_roi_margins[0], baseline_margin_m=cfg.person_roi_margins[1])
                    people = self.people.process(paths, infos, ids, num_frames=frames, polygons=polygons)
                    store.save("people", people_identity, {"observations": people})
                    statuses["person_observations"] = "executed"
                else:
                    row = cached["observations"]
                    people = ObjectObservations(**{**row, "camera_ids": tuple(row["camera_ids"]), "size": tuple(row["size"])})
                    statuses["person_observations"] = "loaded"
            if people.camera_ids != ids or people.num_frames != frames or people.size != (info.width, info.height) or abs(people.fps - info.fps) > 1e-6 or people.uv_px.shape[-2] != 17:
                raise ValueError("Person observations do not match the source camera/timeline contract")
            timings["person_observations"] = time.monotonic() - tick

            tick = time.monotonic()
            self.last_receipt["active_stage"] = "ball_detection"
            balls = self._empty_objects(ids, info, frames, 1)
            if self.ball is not None:
                ball_identity = {**common, "settings": cfg.processing_settings["ball_detection"], "checkpoint": self._file_identity(cfg.ball_detection.checkpoint)}
                cached = store.load("ball", ball_identity)
                if cached is None:
                    try:
                        detected = self.ball.process(paths, max_frames=max_frames, image_width=info.width, image_height=info.height)
                    finally:
                        self.ball.unload()
                    store.save("ball", ball_identity, {"ball_uv": detected.ball_uv, "ball_uv_px": detected.ball_uv_px, "visibility": detected.visibility, "score": detected.score})
                    statuses["ball_detection"] = "executed"
                else:
                    detected = BallDetectionResult.from_dict(cached)
                    statuses["ball_detection"] = "loaded"
                valid, errors = detected.validate()
                if not valid or detected.ball_uv.shape != (len(ids), frames, 2):
                    raise ValueError(f"Ball observations violate the source timeline: {errors}")
                balls = ObjectObservations(ids, (info.width, info.height), info.fps, detected.ball_uv_px[:, :, None, None], detected.score[:, :, None, None].astype(np.float32), detected.visibility[:, :, None], np.zeros((len(ids), 1), np.int64))
            timings["ball_detection"] = time.monotonic() - tick
            has_people = self.plcs is not None and people.visibility(cfg.human_vis_threshold).any()
            has_ball = self.blcs is not None and balls.visibility(cfg.ball_detection.score_threshold).any()
            if not has_people and not has_ball:
                for stage in ("plcs_association", "blcs_association", "camera_geometry", "player_reconstruction", "ball_reconstruction", "gvhmr"):
                    if cfg.enabled[stage]:
                        statuses[stage] = "skipped_no_observations"
                scene = assemble_automatic_scene(video_paths=paths, camera_ids=ids, info=info, court=court, active_indices=(), geometry=None, grouped_people=None, skeleton=None, players=None, ball=None,
                    metadata={"status": "empty", "enabled_stages": [s.value for s in self.execution_order], "stage_status": statuses, "artifacts": store.references, "source_frame_indices": sample_frames.tolist(), "source_identity": source, "code_sha256": self.code_identity})
            else:
                reference = calibration.reference(cfg.camera_geometry)
                active = tuple(local.source_index for local in calibration.views)
                selected_people, selected_balls = people.select_views(active), balls.select_views(active)
                predictions: list[AssociationObservationResult] = []
                evidence: list[SideEvidence] = []
                scale = np.hypot(info.width, info.height) / np.hypot(1920, 1080)
                tick = time.monotonic()
                p_result, p_group = self._associate(self.plcs, "plcs", selected_people, court, active, sample_frames, reference, store, common, statuses)
                timings["plcs_association"] = time.monotonic() - tick
                if p_result is not None:
                    predictions.append(p_result)
                    torso = [5, 6, 11, 12]
                    evidence.append(SideEvidence("plcs", p_group.uv_px[:, :, sample_frames][:, :, :, torso], (p_group.visibility & (p_group.confidence >= cfg.joint_confidence))[:, :, sample_frames][:, :, :, torso], cfg.player_reprojection_px * scale))
                tick = time.monotonic()
                b_result, b_group = self._associate(self.blcs, "blcs", selected_balls, court, active, sample_frames, reference, store, common, statuses)
                timings["blcs_association"] = time.monotonic() - tick
                if b_result is not None:
                    predictions.append(b_result)
                    evidence.append(SideEvidence("blcs", b_group.uv_px[:, :, sample_frames], b_group.visibility[:, :, sample_frames], cfg.ball_reprojection_px * scale))
                if not predictions:
                    raise ReconstructionUnavailable("no_observations_in_calibrated_views", "No model input in accepted camera views")
                tick = time.monotonic()
                self.last_receipt["active_stage"] = "camera_geometry"
                geometry = resolve_camera_geometry(calibration, reference, tuple(p.view_half_turns.numpy() for p in predictions), tuple(evidence), config=cfg.camera_geometry)
                statuses["camera_geometry"] = "executed"
                timings["camera_geometry"] = time.monotonic() - tick
                reconstruction_identity = {**common, "associations": {k: v for k, v in store.references.items() if k.endswith("_association")}, "geometry": geometry.document,
                    "settings": {key: cfg.processing_settings[key] for key in ("player_reconstruction", "ball_reconstruction", "gvhmr")}}
                tick = time.monotonic()
                self.last_receipt["active_stage"] = "triangulation"
                cached = store.load("triangulation", reconstruction_identity)
                if cached is None:
                    skeleton = triangulate_players(p_group, geometry.cameras, reprojection_px=cfg.player_reprojection_px * scale, joint_confidence=cfg.joint_confidence) if cfg.enabled["player_reconstruction"] else None
                    ball = reconstruct_rally_ball(b_group, geometry.cameras, fps=info.fps, reprojection_px=cfg.ball_reprojection_px * scale, min_frames=cfg.ball_min_frames, ambiguity_ratio=cfg.ball_ambiguity_ratio) if cfg.enabled["ball_reconstruction"] else None
                    store.save("triangulation", reconstruction_identity, {"skeleton": skeleton, "ball": ball})
                else:
                    skeleton = None if cached["skeleton"] is None else PlayerSkeleton(**cached["skeleton"])
                    b = cached["ball"]
                    ball = None if b is None else BallReconstructionResult(**{**b, "trajectory": TriangulatedPoints(**b["trajectory"]), "candidate_counts": {int(k): int(v) for k, v in b["candidate_counts"].items()}})
                if cfg.enabled["player_reconstruction"]:
                    statuses["player_reconstruction"] = "ok" if skeleton is not None and skeleton.valid.any() else "insufficient_support"
                if cfg.enabled["ball_reconstruction"]:
                    statuses["ball_reconstruction"] = "insufficient_support" if ball is None else ball.status
                timings["triangulation"] = time.monotonic() - tick
                tick = time.monotonic()
                self.last_receipt["active_stage"] = "gvhmr"
                players = None
                if skeleton is not None:
                    body_identity = {**reconstruction_identity, "triangulation": store.references["triangulation"], "models": {
                        "hmr2": self._file_identity(cfg.people.hmr2_checkpoint), "gvhmr": self._file_identity(cfg.people.gvhmr_checkpoint),
                        "root_regressor": self._file_identity(cfg.people.bundled_assets.smpl_neutral_joint_regressor),
                        "smplx": self._file_identity(cfg.people.body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz"),
                        "topology": self._file_identity(cfg.people.bundled_assets.smplx_to_smpl),
                        "coco17": self._file_identity(cfg.people.bundled_assets.smpl_coco17_regressor)},
                        "runtime": json_value(cfg.people.runtime)}
                    cached = store.load("bodies", body_identity)
                    if cached is None:
                        placement_config = replace(cfg.player_placement,
                            max_reprojection_rms_px=cfg.player_placement.max_reprojection_rms_px * scale,
                            reprojection_weight_sigma_px=cfg.player_placement.reprojection_weight_sigma_px * scale)
                        players = reconstruct_player_bodies(p_group, selected_people, skeleton, geometry.cameras, tuple(paths[i] for i in active), sample_frames,
                            body=self.body, reprojection_px=cfg.player_reprojection_px * scale, placement_config=placement_config)
                        store.save("bodies", body_identity, {"players": players})
                    else:
                        players = ReconstructedPlayers(**cached["players"])
                    if cfg.enabled["gvhmr"]:
                        statuses["gvhmr"] = "ok" if players.smpl_valid.any() else "insufficient_support"
                timings["gvhmr"] = time.monotonic() - tick
                has_3d = (skeleton is not None and skeleton.valid.any()) or (ball is not None and ball.trajectory.valid.any())
                if not has_3d:
                    raise ReconstructionUnavailable("no_valid_reconstruction", "Nonempty observations produced no supported 3D geometry")
                required = [key for key in ("player_reconstruction", "ball_reconstruction", "gvhmr") if cfg.enabled[key]]
                status = "ok" if all(statuses.get(key) == "ok" for key in required) else "partial"
                scene = assemble_automatic_scene(video_paths=paths, camera_ids=ids, info=info, court=court, active_indices=active, geometry=geometry, grouped_people=p_group, skeleton=skeleton, players=players, ball=ball,
                    metadata={"status": status, "enabled_stages": [s.value for s in self.execution_order], "stage_status": statuses, "artifacts": store.references,
                        "source_frame_indices": sample_frames.tolist(), "source_identity": source, "code_sha256": self.code_identity,
                        "side_logits": {task: result.side_logits.tolist() for task, result in (("plcs", p_result), ("blcs", b_result)) if result is not None}})
            self.last_receipt["active_stage"] = None
            self.last_receipt.update(status=scene.metadata["status"], artifacts=store.references, validity=scene.metadata["validity_statistics"])
            timings["total"] = time.monotonic() - started
            scene.metadata["stage_seconds"] = dict(timings)
            write_json_atomic(store.root / "run.json", self.last_receipt)
            return scene
        except Exception as exc:
            active_stage = self.last_receipt.get("active_stage")
            if isinstance(active_stage, str):
                statuses[active_stage] = "failed"
            reason = exc.reason if isinstance(exc, ReconstructionUnavailable) else "contract_or_execution_error"
            diagnostics = exc.diagnostics if isinstance(exc, ReconstructionUnavailable) else {}
            self.last_receipt.update(status="failed", reason=reason, error=str(exc), diagnostics=diagnostics, artifacts=store.references)
            timings["total"] = time.monotonic() - started
            write_json_atomic(store.root / "run.json", self.last_receipt)
            LOGGER.error("Automatic reconstruction failed (%s): %s", reason, exc)
            raise
