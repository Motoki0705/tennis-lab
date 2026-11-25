"""Produce meshes for visual review from frozen GVHMR outputs and FastHMR PT16."""

import json
import os
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from src.submodules.configuration import SmplAssetPaths
from src.submodules.models import SmplVertexReconstructor
from src.submodules.models.fasthmr.assets import FastHMRConfig
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.evaluation.player_pose.bundle import (
    load_body_input_manifest,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.reference import (
    load_regressors,
    scene_cameras,
)
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.body_types import BodyGeometry, BodyRecoveryRequest
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodySelection,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.fasthmr import FastHMRInput, FastHMRModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput
from src.tennis_scene.pipeline.components.hmr2_features import validate_body_selection
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlayerSkeleton,
    reconstruct_player_bodies,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.model_assets import file_identity
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.tennis_scene.schema import validate_scene_result_arrays
from src.tennis_scene.visualization.rendering.smpl_court import smpl_vertices_to_court
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera

DATA = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks")
COHORT = DATA / "multiview-player-evaluation-v1"
SOURCE = (DATA / "multiview-v1").resolve()
COHORT_SHA = "3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580"
COHORT_ID = "meiji_4cam/multiview-player-evaluation-v1"
SMPL_GVHMR = Path("ckpt/body_models/gvhmr").resolve()
FEATURE_BATCH = 4
SEED = 0


def emit(report):
    print(json.dumps(report, allow_nan=False), flush=True)


def cohort_records():
    manifest = load_body_input_manifest(COHORT, expected_sha256=COHORT_SHA)
    source_path = pinned_file(
        SOURCE,
        {
            "path": "manifest.json",
            "sha256": manifest["source_benchmark"]["manifest_sha256"],
        },
    )
    if manifest["cohort_id"] != COHORT_ID:
        raise ValueError("Unexpected body cohort")
    sources = {
        row["clip_id"]: row for row in json.loads(source_path.read_text())["clips"]
    }
    return [(row, sources[row["clip_id"]]) for row in manifest["clips"]]


PT16 = "pt16"


def pt16_config():
    return FastHMRConfig(
        Path("third_party/FastHMR").resolve(),
        Path("ckpt/fasthmr").resolve(),
        SMPL_GVHMR / "smpl_coco17_J_regressor.pt",
        "cuda:0",
        PT16,
        FEATURE_BATCH,
        SEED,
    )


def cuda(backend):
    name = torch.cuda.get_device_name(0)
    if "5060 Ti" not in name:
        raise RuntimeError("This run requires the local RTX 5060 Ti")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    emit(
        {
            "phase": "device",
            "backend": backend,
            "gpu": name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    )


@dataclass(frozen=True)
class Clip:
    record: dict
    fixed: dict
    inputs: FastHMRInput
    store: StoreReader
    cameras: tuple[PinholeCamera, ...]

    @classmethod
    def read(cls, record, source_record):
        source = ClipSource(
            record["clip_id"],
            tuple(
                SourceVideo(
                    video["camera_id"],
                    pinned_file(SOURCE, video),
                    video["sha256"],
                    video["num_frames"],
                    video["fps"],
                    record["width"],
                    record["height"],
                )
                for video in record["videos"]
            ),
        )
        index = StoreReader(pinned_file(SOURCE, source_record["scene"]))
        if (
            index.source["clip_id"] != source.clip_id
            or index.camera_ids != source.camera_ids
            or index.source_sha256 != source_record["scene"]["source_sha256"]
        ):
            raise ValueError("Saved clip index does not match the fixed input")
        with np.load(
            pinned_file(COHORT, record["fixed_inputs"]), allow_pickle=False
        ) as arrays:
            fixed = dict(arrays)
        ids = source.camera_ids
        if (
            not np.array_equal(fixed["camera_ids"], record["camera_ids"])
            or not np.array_equal(fixed["camera_ids"], ids)
            or not np.array_equal(fixed["player_ids"], record["player_ids"])
            or not np.array_equal(
                fixed["evaluation_frames"], record["evaluation_frames"]
            )
            or not np.array_equal(
                fixed["evaluation_frames"],
                np.flatnonzero((fixed["segment_index"] >= 0).any(0)),
            )
        ):
            raise ValueError(
                "Fixed camera, person or frame axis does not match the manifest"
            )
        cameras = tuple(
            PinholeCamera(*row)
            for row in zip(ids, fixed["K"], fixed["R"], fixed["t"], strict=True)
        )
        selections = []
        for row, identity in enumerate(fixed["player_ids"]):
            labels = fixed["segment_index"][row]
            segments = np.unique(labels[labels >= 0])
            view = int(fixed["selected_view_index"][row])
            if not (0 <= view < len(ids)):
                raise ValueError(f"Missing selected camera for person {identity}")
            if not np.array_equal(
                fixed["selected_intrinsic"][row], cameras[view].intrinsic
            ):
                raise ValueError("Selected intrinsic differs from the fixed camera")
            video = source.video(ids[view])
            frames = [np.flatnonzero(labels == segment) for segment in segments]
            requests = tuple(
                BodyRecoveryRequest(
                    video.path,
                    group,
                    fixed["keypoints_xyc"][row, group],
                    fixed["boxes_xys"][row, group],
                    source.size,
                    fixed["selected_intrinsic"][row],
                )
                for group in frames
            )
            counts = tuple(
                int(fixed["selected_view_observed"][row, group].sum())
                for group in frames
            )
            selections.append(BodySelection(int(identity), ids[view], requests, counts))
        inputs = FastHMRInput(source, BodyViewSelectionOutput(tuple(selections)))
        validate_body_selection(source, inputs.selection)
        return cls(record, fixed, inputs, index, cameras)

    def payload(self, name):
        pin = self.record["source_components"][name]
        ref = self.store.references[name]
        if any(
            pin[key] != ref[key]
            for key in ("artifact_id", "execution_key", "sha256", "schema", "version")
        ):
            raise ValueError(f"{name} lineage differs from the pin")
        if self.store.root / ref["path"] != SOURCE / pin["path"]:
            raise ValueError(f"{name} path differs from the pin")
        location = Path(pin["path"]).parent
        payload, descriptor = self.store.payload(name)
        arrays = {
            (location / filename).as_posix(): info["sha256"]
            for filename, info in descriptor["arrays"].items()
        }
        if arrays != pin["arrays"]:
            raise ValueError(f"{name} array hashes differ from the pin")
        return payload

    def tracked_people(self):
        people = self.payload("multiview_scene_inputs")["people"]
        grouped = restore_type(people["grouped"], GroupedObservations)
        raw = restore_type(people["raw"], ObjectObservations)
        skeleton = restore_type(people["skeleton"], PlayerSkeleton)
        pins = {
            "positions": "triangulated_joints_m",
            "valid": "triangulated_valid",
            "reasons": "triangulated_rejection_code",
            "inliers": "triangulated_inliers",
            "reprojection_px": "triangulated_reprojection_px",
        }
        if (
            raw.camera_ids != self.inputs.source.camera_ids
            or raw.fps != self.inputs.source.fps
            or not np.array_equal(grouped.identities, self.fixed["player_ids"])
            or any(
                not np.array_equal(getattr(skeleton, key), self.fixed[value])
                for key, value in pins.items()
            )
        ):
            raise ValueError(
                "Adopted observations differ from the fixed placement inputs"
            )
        return grouped, raw, skeleton

    def baseline_path(self):
        exports = self.record["scene_exports"]
        path = pinned_file(SOURCE, exports["scene/scene"])
        if pinned_file(SOURCE, exports["scene/metadata"]) != path.with_suffix(
            ".metadata.json"
        ):
            raise ValueError("Unexpected scene metadata path")
        return path

    def baseline(self):
        scene = load_scene_result(self.baseline_path())
        if (
            not np.array_equal(scene.player_track_ids, self.fixed["player_ids"])
            or not np.array_equal(
                scene.player_kp_3d, self.fixed["triangulated_joints_m"]
            )
            or any(
                not np.array_equal(a.matrix, b.matrix)
                for a, b in zip(scene_cameras(scene), self.cameras, strict=True)
            )
        ):
            raise ValueError(
                "Scene differs from the pinned person/camera/skeleton axes"
            )
        return scene

    def placement_settings(self):
        return self.store.descriptor("body_placement")["identity"]["settings"]


def camera_arrays(clip, bodies, geometry):
    fixed = clip.fixed
    frames = fixed["evaluation_frames"]
    shape = (len(fixed["player_ids"]), len(frames))
    present = fixed["segment_index"][:, frames] >= 0
    width = bodies[0].segments[0].parameters.pose_width
    arrays = {
        "camera_vertices_m": np.zeros((*shape, 6890, 3), np.float32),
        "camera_coco17_m": np.zeros((*shape, 17, 3), np.float32),
        "camera_valid": np.zeros(shape, bool),
        **{
            key: np.zeros((*shape, dim), np.float32)
            for key, dim in (
                ("body_pose", width),
                ("betas", 10),
                ("global_orient", 3),
                ("transl", 3),
            )
        },
    }
    for player, (body, selection) in enumerate(
        zip(bodies, clip.inputs.selection.selections, strict=True)
    ):
        if (body.person_id, body.camera_id) != (
            selection.person_id,
            selection.camera_id,
        ):
            raise ValueError("Predictions disagree with selected player or view")
        for segment, request, count in zip(
            body.segments, selection.requests, selection.observed_samples, strict=True
        ):
            params = segment.parameters
            if (
                not np.array_equal(segment.source_frames, request.source_frames)
                or segment.observed_samples != count
            ):
                raise ValueError("Predictions disagree with selected segment")
            rows = np.searchsorted(frames, segment.source_frames)
            for key in ("body_pose", "betas", "global_orient", "transl"):
                arrays[key][player, rows] = getattr(params, key)
            for offset in range(0, len(rows), 32):
                batch = slice(offset, offset + 32)
                mesh = geometry(
                    type(params)(
                        *(
                            getattr(params, field.name)[batch]
                            for field in fields(params)
                        )
                    )
                )
                expected = (len(rows[batch]),)
                if (
                    mesh.vertices.shape != (*expected, 6890, 3)
                    or mesh.coco17.shape != (*expected, 17, 3)
                    or not np.isfinite(mesh.vertices).all()
                    or not np.isfinite(mesh.coco17).all()
                ):
                    raise ValueError("Invalid camera-relative mesh or joints")
                arrays["camera_vertices_m"][player, rows[batch]] = mesh.vertices
                arrays["camera_coco17_m"][player, rows[batch]] = mesh.coco17
            arrays["camera_valid"][player, rows] = True
    if not np.array_equal(arrays["camera_valid"], present):
        raise ValueError("Camera meshes did not preserve every fixed inference frame")
    return arrays


def placed_arrays(clip, scene, root, coco):
    frames = clip.fixed["evaluation_frames"]
    valid = scene.player_smpl_valid[:, frames]
    world = np.zeros((*valid.shape, 6890, 3), np.float32)
    if valid.any():
        world = smpl_vertices_to_court(
            scene.smpl_vertices_local[:, frames],
            scene.smpl_global_orient[:, frames],
            scene.player_position[:, frames],
            scene.player_yaw[:, frames],
            root.reshape(1, -1),
        )
        world[~valid] = 0
    return {
        "court_vertices_m": world,
        "court_coco17_m": np.einsum("jv,ptvc->ptjc", coco, world),
        "court_valid": valid,
        "root_valid": scene.player_valid[:, frames],
        "heading_valid": scene.player_heading_valid[:, frames],
        "placement_root_reason": scene.player_rejection_code[:, frames],
    }


def output_clip(output, mode, clip, before, after, diagnostics):
    frames = clip.fixed["evaluation_frames"]
    ids = clip.fixed["player_ids"]
    segment = clip.fixed["segment_index"][:, frames]
    layout = {
        "source_frames": frames,
        "player_ids": ids,
        "camera_ids": clip.fixed["camera_ids"],
        "selected_view_index": clip.fixed["selected_view_index"],
        "segment_index": segment,
        "selected_view_observed": clip.fixed["selected_view_observed"][:, frames],
        "K": clip.fixed["K"],
        "R": clip.fixed["R"],
        "t": clip.fixed["t"],
    }
    if mode == PT16:
        width, height = clip.inputs.source.size
        focal = np.hypot(width, height)
        intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        layout["camera_mesh_K"] = np.broadcast_to(intrinsic, (len(ids), 3, 3))
    else:
        layout["camera_mesh_K"] = clip.fixed["selected_intrinsic"]
    name = clip.inputs.source.clip_id.split("/")[-1]
    path = output / f"{name}-{mode}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **layout, **before, **after)
    temporary.replace(path)
    return {
        "phase": "clip",
        "status": "ok",
        "backend": mode,
        "clip_id": clip.inputs.source.clip_id,
        "frames": len(frames),
        "npz": path.name,
        "sha256": dual_sha256(path),
        "placement": diagnostics,
        "players": [
            {
                "id": int(identity),
                "camera": str(layout["camera_ids"][layout["selected_view_index"][row]]),
                "camera_mesh_frames": int(before["camera_valid"][row].sum()),
                "court_mesh_frames": int(after["court_valid"][row].sum()),
                "court_mesh_frames_at_samples": int(
                    (before["camera_valid"][row] & after["court_valid"][row]).sum()
                ),
                "input_absent_frames": int((segment[row] < 0).sum()),
                "input_frames_with_interpolated_crop": int(
                    ((segment[row] >= 0) & ~layout["selected_view_observed"][row]).sum()
                ),
            }
            for row, identity in enumerate(ids)
        ],
    }


def baseline_assets(clip):
    model = clip.placement_settings()["models"]["body"]["assets"]
    paths = {}
    for key, pin in model.items():
        path = (
            Path("ckpt/body_models/smplx") if key == "body_model" else SMPL_GVHMR
        ) / Path(pin["path"]).name
        if file_identity(path)["sha256"] != pin["sha256"]:
            raise ValueError(f"GVHMR geometry asset differs from baseline: {key}")
        paths[key] = path.resolve()
    return model, paths


def baseline_geometry(clip):
    model, paths = baseline_assets(clip)
    args = {field.name: paths[f"smpl_{field.name}"] for field in fields(SmplAssetPaths)}
    assets = SmplAssetPaths(**args)
    mesh = SmplVertexReconstructor(
        paths["body_model"].parents[1], device="cuda:0", smpl_assets=assets
    )
    regressors = load_regressors(SMPL_GVHMR)

    def geometry(params):
        vertices = mesh.reconstruct(params.tensors()).numpy()
        return BodyGeometry(
            vertices, np.einsum("jv,tvc->tjc", regressors.coco17, vertices)
        )

    return model, geometry, regressors


def baseline_clip(output, clip, models):
    if models[0] != clip.placement_settings()["models"]["body"]["assets"]:
        raise ValueError("Baseline body assets differ across clips")
    _, reconstruct, regressors = models
    scene = clip.baseline()
    recovered = restore_type(clip.payload("gvhmr"), GVHMROutput)
    before = camera_arrays(clip, recovered.bodies, reconstruct)
    after = placed_arrays(clip, scene, regressors.root, regressors.coco17)
    return output_clip(
        output, "gvhmr", clip, before, after, scene.metadata["body_placement"]["config"]
    )


def candidate_clip(output, clip, module, models):
    scene = clip.baseline()
    grouped, raw, skeleton = clip.tracked_people()
    settings = clip.placement_settings()
    config = TemporalPlacementConfig(**scene.metadata["body_placement"]["config"])
    if as_settings := {
        key: value
        for key, value in settings["placement"].items()
        if getattr(config, key)
        != (
            value * clip.inputs.source.pixel_threshold_scale
            if key in {"max_reprojection_rms_px", "reprojection_weight_sigma_px"}
            else value
        )
    }:
        raise ValueError(f"Placement settings disagree with saved scene: {as_settings}")
    result = module.process_with_models(clip.inputs, models)
    body = models.borrow("body")
    before = camera_arrays(clip, result.bodies, body.reconstruct)
    players = reconstruct_player_bodies(
        grouped,
        raw,
        skeleton,
        clip.cameras,
        result,
        body=body,
        placement_config=config,
        reprojection_px=settings["reprojection_px"]
        * clip.inputs.source.pixel_threshold_scale,
    )
    placed_scene = replace(
        scene,
        player_position=players.position,
        player_yaw=players.yaw,
        player_valid=players.root_valid,
        player_heading_valid=players.heading_valid,
        player_rejection_code=players.root_reasons,
        player_smpl_valid=players.smpl_valid,
        smpl_body_pose=players.body_pose,
        smpl_global_orient=players.global_orient,
        smpl_betas=players.betas,
        smpl_vertices_local=players.vertices_local,
        metadata={
            **scene.metadata,
            "body_placement": players.diagnostics,
            "body_recovery": "fasthmr",
        },
    )
    validate_scene_result_arrays(placed_scene)
    after = placed_arrays(
        clip,
        placed_scene,
        body.root_regressor.astype(np.float32),
        body.coco17.cpu().numpy(),
    )
    return output_clip(output, PT16, clip, before, after, players.diagnostics["config"])


def run(mode):
    records = cohort_records()
    if mode == "check-inputs":
        for record, source in records:
            clip = Clip.read(record, source)
            clip.tracked_people()
            baseline = restore_type(clip.payload("gvhmr"), GVHMROutput)
            for body, selection in zip(
                baseline.bodies, clip.inputs.selection.selections, strict=True
            ):
                if (body.person_id, body.camera_id) != (
                    selection.person_id,
                    selection.camera_id,
                ):
                    raise ValueError("Saved GVHMR body axis differs from the freeze")
                for segment, request in zip(
                    body.segments, selection.requests, strict=True
                ):
                    np.testing.assert_array_equal(
                        segment.source_frames, request.source_frames
                    )
            baseline_assets(clip)
            with np.load(clip.baseline_path(), allow_pickle=False) as scene:
                shape = (len(clip.fixed["player_ids"]), clip.inputs.source.num_frames)
                required = {
                    "smpl_vertices_local",
                    "smpl_global_orient",
                    "player_position",
                    "player_yaw",
                }
                if (
                    not required.issubset(scene.files)
                    or scene["player_smpl_valid"].shape != shape
                ):
                    raise ValueError(
                        "Saved GVHMR scene has no usable placement geometry"
                    )
                if not np.array_equal(
                    scene["player_track_ids"], clip.fixed["player_ids"]
                ):
                    raise ValueError("Saved GVHMR scene has a different player axis")
            emit(
                {
                    "phase": "input",
                    "clip": clip.inputs.source.clip_id,
                    "frames": len(clip.fixed["evaluation_frames"]),
                    "baseline": "archived_gvhmr_parameters_and_placed_scene",
                }
            )
        return 0

    cuda(mode)
    output = Path(os.environ["TENNIS_REPRO_DIR"]) / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    module = None if mode == "gvhmr-archive" else FastHMRModule(pt16_config())
    operations = (
        {}
        if module is None
        else {
            (record["clip_id"], "fasthmr"): module.model_specs()
            for record, _ in records
        }
    )
    reports = []
    with InferenceRuntime(operations) as runtime:
        models = None
        for record, source in records:
            emit({"phase": "start_clip", "backend": mode, "clip": record["clip_id"]})
            try:
                clip = Clip.read(record, source)
                if module is None:
                    if models is None:
                        models = baseline_geometry(clip)
                    report = baseline_clip(output, clip, models)
                else:
                    with runtime.node(
                        (clip.inputs.source.clip_id, "fasthmr")
                    ) as borrower:
                        report = candidate_clip(output, clip, module, borrower)
            except Exception as exc:
                traceback.print_exc()
                report = {
                    "phase": "clip",
                    "status": "error",
                    "backend": mode,
                    "clip_id": record["clip_id"],
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            reports.append(report)
            emit(report)
    (output / f"review-{mode}.json").write_text(
        json.dumps(
            {
                "purpose": "qualitative_visualization",
                "cohort_id": COHORT_ID,
                "cohort_sha256": COHORT_SHA,
                "backend": mode,
                "feature_batch_size": FEATURE_BATCH if module else None,
                "seed": SEED if module else None,
                "clips": reports,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return int(any(row["status"] != "ok" for row in reports))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-inputs", "gvhmr-archive", PT16))
    raise SystemExit(run(parser.parse_args().mode))
