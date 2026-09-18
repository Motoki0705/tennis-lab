"""Read-only dataset audit; diagnostic tails never silently exclude clips."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import exact_config_mapping
from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tasks.slcs.data.dino_tokens import load_dino_tokens
from src.tasks.slcs.data.quality import QualityConfig, build_label_masks
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.dataset_pipeline.quality import evaluate_reconstruction, summarize
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.reference_pipeline.observations import sha256
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.io import save_json_atomic
from src.utils.paths import PROJECT_ROOT


def validate_quality_report_config(cfg: DictConfig) -> None:
    """Validate the CLI contract without opening datasets, media or artifacts."""
    exact_config_mapping(
        cfg,
        path="quality_report",
        required_keys={
            "paths",
            "dataset_directory",
            "expected_dataset_directory",
            "generation_directories",
            "observation_directories",
            "excluded_clips",
            "allow_incomplete",
            "quality",
            "output_dir",
        },
    )
    if type(cfg.allow_incomplete) is not bool:
        raise ValueError("allow_incomplete must be a boolean")
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(dict(cfg.paths), repository_root=PROJECT_ROOT)
    )
    for name, role in (
        ("dataset_directory", PathRole.DATA),
        ("expected_dataset_directory", PathRole.DATA),
        ("output_dir", PathRole.OUTPUT),
    ):
        value = cfg[name]
        if type(value) is not str:
            raise ValueError(f"{name} must be a role-relative string")
        resolver.resolve(role, value)
    for name in ("generation_directories", "observation_directories"):
        values = cfg[name]
        if not OmegaConf.is_list(values) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        if any(type(value) is not str for value in values) or len(set(values)) != len(
            values
        ):
            raise ValueError(f"{name} must contain unique role-relative strings")
        for value in values:
            resolver.resolve(PathRole.OUTPUT, value)
    if not isinstance(cfg.excluded_clips, DictConfig):
        raise ValueError("excluded_clips must be a clip-ID to reason mapping")
    for key, reason in cfg.excluded_clips.items():
        if type(key) is not str or len(key.split("/")) != 2:
            raise ValueError("Exclusion IDs must be video_id/clip_id")
        from src.tennis_scene.generate_dataset.manifest import validate_id_component

        for part in key.split("/"):
            validate_id_component(part, field_name="excluded_clips")
        if type(reason) is not str or not reason.strip():
            raise ValueError("Every exclusion requires a nonempty reason")
    exact_config_mapping(
        cfg.quality,
        path="quality",
        required_keys={
            "min_player_confidence",
            "min_ball_cameras",
            "label_weight_power",
            "min_window_label_ratio",
        },
    )
    if type(cfg.quality.min_ball_cameras) is not int:
        raise ValueError("quality.min_ball_cameras must be an integer")
    for name in (
        "min_player_confidence",
        "label_weight_power",
        "min_window_label_ratio",
    ):
        value = cfg.quality[name]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"quality.{name} must be a finite number")
    QualityConfig(**dict(cfg.quality))


def longest_gap(mask: np.ndarray) -> int:
    """Longest unsupported contiguous run, including clip boundaries."""
    edges = np.diff(np.r_[False, ~np.asarray(mask, bool), False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(lengths.max()) if lengths.size else 0


def trajectory_metrics(
    scene: SceneResult,
    player: np.ndarray,
    ball: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Both endpoints must be supported: never differentiate across a gap."""
    assert scene.ball_3d is not None
    ps = np.linalg.norm(np.diff(scene.player_position, axis=1), axis=-1) * scene.fps
    bs = np.linalg.norm(np.diff(scene.ball_3d, axis=0), axis=-1) * scene.fps
    height = scene.ball_3d[:, 2][ball]
    return {
        "player_speed_mps": summarize(ps[player[:, :-1] & player[:, 1:]]),
        "ball_speed_mps": summarize(bs[ball[:-1] & ball[1:]]),
        "ball_height_m": summarize(height),
        "ball_negative_height_fraction_below_minus_0_1m": float((height < -0.1).mean())
        if height.size
        else None,
        "ball_reprojection_px": summarize(
            np.where(ball[None], arrays["ball_reprojection_px"], np.nan)
        ),
        "pose_all_joint_reprojection_px": summarize(
            np.where(player[:, None, :, None], arrays["pose_reprojection_px"], np.nan)
        ),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _locate(roots: list[Path], key: str, filename: str) -> Path:
    candidates = [root / key for root in roots if (root / key / filename).is_file()]
    if not candidates:
        raise FileNotFoundError(f"Missing {key}/{filename} in explicit input roots")
    hashes = {sha256(path / filename) for path in candidates}
    if len(hashes) != 1:
        raise ValueError(f"Conflicting copies of {key}/{filename}: {candidates}")
    return candidates[0]


def _digest(value: object) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _check_digest(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"Missing/invalid SHA-256: {name}")


def validate_input_identity(
    clip: ClipManifest, identity: dict[str, Any], source_manifest_sha256: str
) -> None:
    """Validate producer inputs against source manifest and actual target bytes."""
    if identity["clip_manifest_sha256"] != source_manifest_sha256:
        raise ValueError("Teacher source manifest identity mismatch")
    actual = {camera: sha256(clip.media_path(camera)) for camera in clip.camera_ids}
    if identity["video_sha256"] != actual:
        raise ValueError("Teacher video identity differs from actual media bytes")


def validate_raw_identity(
    raw: SceneResult, refined: SceneResult, identity: dict[str, Any]
) -> None:
    """Validate available raw provenance and exact observed inputs, not invented receipts."""
    for task, digest in identity["checkpoints"].items():
        if raw.metadata["checkpoints"][task]["sha256"] != digest:
            raise ValueError(f"Raw teacher checkpoint mismatch: {task}")
    for key in ("reference", "ball_input_provenance"):
        if raw.metadata[key] != refined.metadata[key]:
            raise ValueError(f"Raw observation provenance mismatch: {key}")
    for name in (
        "court_kp",
        "court_vis",
        "human_kp_2d",
        "human_kp_vis",
        "ball_uv",
        "ball_vis",
    ):
        if not np.array_equal(getattr(raw, name), getattr(refined, name)):
            raise ValueError(f"Raw observation arrays differ: {name}")


def observation_producers(
    directory: Path, cameras: list[str], observation_hashes: dict[str, str]
) -> dict[str, Any]:
    """Compare recorded producers; this does not authenticate checkpoint files."""
    for name, digest in observation_hashes.items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Observation identity mismatch: {name}")
    people = []
    for camera in cameras:
        name = f"{camera}_people.metadata.json"
        if name not in observation_hashes:
            raise ValueError(f"Unbound people producer receipt: {name}")
        receipt = _json(directory / name)
        if type(receipt["schema_version"]) is not int or receipt["schema_version"] < 1:
            raise ValueError("Invalid people producer schema_version")
        if not isinstance(receipt["policy"], str) or not receipt["policy"].strip():
            raise ValueError("Missing people producer policy")
        if not isinstance(receipt["settings"], dict):
            raise ValueError("Invalid people producer settings")
        for key in ("detector_sha256", "pose_sha256"):
            _check_digest(receipt[key], key)
        people.append(
            {
                "schema_version": receipt["schema_version"],
                "policy": receipt["policy"],
                "settings_sha256": _digest(receipt["settings"]),
                "detector_sha256": receipt["detector_sha256"],
                "pose_sha256": receipt["pose_sha256"],
            }
        )
    if not people or len({_digest(producer) for producer in people}) != 1:
        raise ValueError("Mixed provenance: people_producer across cameras")
    if "court.json" not in observation_hashes:
        raise ValueError("Unbound court producer receipt")
    court = _json(directory / "court.json")["identity"]
    _check_digest(court["checkpoint_sha256"], "court checkpoint")
    if not isinstance(court["settings"], dict):
        raise ValueError("Invalid court producer settings")
    return {
        "people_producer": people[0],
        "court_producer": {
            "checkpoint_sha256": court["checkpoint_sha256"],
            "settings_sha256": _digest(court["settings"]),
        },
    }


def audit_clip(
    clip: ClipManifest,
    key: str,
    runs: list[Path],
    observations: list[Path],
    quality: QualityConfig,
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    scene = load_slcs_annotation(clip)
    if not np.isfinite(scene.fps) or scene.fps <= 0:
        raise ValueError("Invalid fps")
    cameras = list(clip.camera_ids)
    if scene.metadata["reference"]["camera_ids"] != cameras:
        raise ValueError("Scene camera order differs from manifest")
    for name in (
        "player_position",
        "player_yaw",
        "ball_3d",
        "human_kp_2d",
        "ball_uv",
        "human_kp_vis",
        "court_vis",
    ):
        arr = getattr(scene, name)
        if arr is None or not np.isfinite(arr).all():
            raise ValueError(f"Missing/nonfinite {name}")
        if name.endswith("_vis") and ((arr < 0) | (arr > 1)).any():
            raise ValueError(f"Out-of-range {name}")
    identity = scene.metadata["dataset_producer_identity"]
    for task in ("plcs", "blcs"):
        _check_digest(identity["checkpoints"][task], task)
    assert (
        scene.human_kp_vis is not None
        and scene.ball_vis is not None
        and scene.ball_3d is not None
    )
    evidence = scene.metadata["label_quality"]
    masks = build_label_masks(
        human_kp_vis=scene.human_kp_vis,
        ball_vis=scene.ball_vis,
        player_position=scene.player_position,
        player_yaw=scene.player_yaw,
        ball_3d=scene.ball_3d,
        config=quality,
        teacher_quality=evidence,
    )
    validate_input_identity(clip, identity, source_manifest_sha256)
    player = masks["player_label_weight"] > 0
    ball = masks["ball_label_weight"] > 0
    refinement = identity["settings"]["refinement"]
    if (
        min((np.asarray(evidence["player_weight"]) > 0).mean(axis=1))
        < refinement["min_player_label_fraction"]
        or (np.asarray(evidence["ball_weight"]) > 0).mean()
        < refinement["min_ball_label_fraction"]
    ):
        raise ValueError("Teacher coverage violates recorded refinement contract")
    obs = _locate(observations, key, "court.npz")
    producers = observation_producers(obs, cameras, identity["observations"])
    run = _locate(runs, key, "raw_model_quality.json")
    for name in (
        "raw_model_quality.json",
        "quality.json",
        "label_evidence.json",
        "quality_arrays.npz",
    ):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    court = _json(obs / "court.json")
    receipt = _json(obs / "ball_import.metadata.json")
    if receipt["camera_ids"] != cameras:
        raise ValueError("Ball receipt camera order differs")
    statuses = np.asarray(receipt["status"])
    if statuses.shape != (len(cameras), clip.num_frames):
        raise ValueError("Ball status shape differs")
    observed: dict[str, Any] = {}
    for i, camera in enumerate(cameras):
        with np.load(obs / f"{camera}_people.npz", allow_pickle=False) as data:
            observed[camera] = {}
            for name in ("pose_supported_mask", "observed_masks"):
                mask = data[name]
                if mask.shape != player.shape or not np.isin(mask, [0, 1]).all():
                    raise ValueError(f"Invalid {camera}/{name}")
                observed[camera][name] = {
                    "fraction": mask.mean(axis=1).tolist(),
                    "longest_gap_frames": [longest_gap(m) for m in mask],
                }
        values, counts = np.unique(statuses[i], return_counts=True)
        observed[camera]["ball_status_fraction"] = {
            str(v): float(c / clip.num_frames)
            for v, c in zip(values, counts, strict=True)
        }
    feature_marker = _json(clip.clip_dir / "annotations/dino_v3/annotation.json")
    if (
        feature_marker["input_manifest_digest"] != clip.digest()
        or list(feature_marker["cameras"]) != cameras
    ):
        raise ValueError("DINO manifest digest/camera order mismatch")
    if feature_marker["generator"]["video_sha256"] != identity["video_sha256"]:
        raise ValueError("DINO video provenance differs from teacher")
    feature_hash = feature_marker["generator"]["checkpoint_sha256"]
    _check_digest(feature_hash, "DINO")
    specs = []
    expected_spec = None
    for camera in cameras:
        _, _, spec = load_dino_tokens(clip, camera, expected_spec=expected_spec)
        if expected_spec is not None and spec != expected_spec:
            raise ValueError("DINO camera specs differ")
        expected_spec = spec
        specs.append(asdict(spec))
    with np.load(obs / "court.npz", allow_pickle=False) as data:
        homographies = data["homographies"]
    turns = identity["settings"]["view_half_turns"]
    if identity["settings"]["coordinate_mode"] != "reference":
        turns = [False] * len(cameras)
    raw = load_scene_result(run / "scene.npz")
    validate_raw_identity(raw, scene, identity)
    stages = {}
    for stage, current in [
        ("raw", raw),
        ("refined", scene),
    ]:
        if (
            current.num_frames != clip.num_frames
            or current.fps != clip.fps
            or current.metadata["reference"]["camera_ids"] != cameras
        ):
            raise ValueError(f"{stage} run scene disagrees with manifest")
        _, arrays = evaluate_reconstruction(current, homographies, turns)
        if stage == "refined":
            with np.load(run / "quality_arrays.npz", allow_pickle=False) as saved:
                for name, values in arrays.items():
                    if saved[name].shape != values.shape or not np.allclose(
                        saved[name], values, equal_nan=True
                    ):
                        raise ValueError(
                            f"Run diagnostics disagree with dataset scene: {name}"
                        )
        stages[stage] = {
            "all_trajectory": trajectory_metrics(
                current, np.ones_like(player), np.ones_like(ball), arrays
            ),
            "slcs_positive_weight": trajectory_metrics(current, player, ball, arrays),
        }
    sources = {}
    for name, mask in [("player", player), ("ball", ball)]:
        source = np.asarray(evidence[f"{name}_source"])
        if source.shape != mask.shape or not np.isin(source, [0, 1, 2, 3]).all():
            raise ValueError(f"Invalid {name} source codes")
        sources[name] = {
            str(code): {
                "all": int((source == code).sum()),
                "slcs_positive_weight": int(((source == code) & mask).sum()),
            }
            for code in range(4)
        }
    return {
        **producers,
        "producer_verification": "recorded producer identities and bound observation bytes only; detector/pose/court checkpoint files are not authenticated",
        "source_manifest_sha256": source_manifest_sha256,
        "manifest_sha256": clip.digest(),
        "raw_provenance_verification": "checkpoint SHA, reference/ball receipt and exact observed arrays verified; raw archive has no complete producer identity, so all execution settings cannot be authenticated",
        "observation_sha256": identity["observations"],
        "run_artifact_sha256": {
            name: sha256(run / name)
            for name in (
                "scene.npz",
                "scene.metadata.json",
                "raw_model_quality.json",
                "quality.json",
                "label_evidence.json",
                "quality_arrays.npz",
            )
        },
        "fps": clip.fps,
        "frames": clip.num_frames,
        "camera_ids": cameras,
        "teacher_checkpoints": identity["checkpoints"],
        "teacher_settings_sha256": _digest(identity["settings"]),
        "dino_checkpoint_sha256": feature_hash,
        "dino_spec": specs[0],
        "scene_sha256": sha256(clip.clip_dir / "annotations/tennis_scene/scene.npz"),
        "run_directory": str(run),
        "observation_directory": str(obs),
        "coverage": {
            "player": player.mean(axis=1).tolist(),
            "ball": float(ball.mean()),
        },
        "unsupported_longest_gap_frames": {
            "player": [longest_gap(m) for m in player],
            "ball": longest_gap(ball),
        },
        "sources": sources,
        "observations": observed,
        "stages": stages,
        "court_image_homography_fit": court["diagnostics"],
        "approximate_pinhole_fit_rmse_px": [
            fit["rmse_px"] for fit in scene.metadata["reference"]["camera_fits"]
        ],
        "review_flags": [
            "observable_consistency_not_measured_3d_accuracy",
            "all_joint_reprojection_has_no_triangulation_acceptance_threshold",
        ],
    }


def write_quality_report(
    dataset: Path,
    expected_dataset: Path,
    runs: list[Path],
    observations: list[Path],
    output: Path,
    *,
    quality: QualityConfig,
    excluded_clips: dict[str, str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Persist JSON and CSV even on failure; snapshot only tolerates missing clips."""
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset),
        "expected_dataset": str(expected_dataset),
        "quality_config": asdict(quality),
        "allow_incomplete": allow_incomplete,
        "clips": {},
        "errors": [],
    }
    excluded = excluded_clips or {}
    try:
        expected = load_dataset_manifest(expected_dataset).clips
        actual = load_dataset_manifest(dataset).clips
        if set(actual) - set(expected) or set(excluded) - set(expected):
            raise ValueError("Unexpected dataset or exclusion clip IDs")
        if set(actual) & set(excluded):
            raise ValueError(
                f"Excluded clips remain in dataset manifest: {sorted(set(actual) & set(excluded))}"
            )
        if any(not reason.strip() for reason in excluded.values()):
            raise ValueError("Exclusions require explicit nonempty reasons")
        for key in sorted(expected):
            row: dict[str, Any] = {"status": "missing"}
            report["clips"][key] = row
            if key in excluded:
                row.update(status="excluded", reason=excluded[key])
                continue
            if key not in actual:
                row["reason"] = "missing dataset manifest record"
                continue
            clip_dir = dataset / actual[key].path
            try:
                source_clip = ClipManifest.load(expected_dataset / expected[key].path)
                target_clip = ClipManifest.load(clip_dir)
                for field in ("num_frames", "fps", "width", "height", "camera_ids"):
                    if getattr(source_clip, field) != getattr(target_clip, field):
                        raise ValueError(f"Source/target manifest mismatch: {field}")
                if not (
                    clip_dir / "annotations/tennis_scene/annotation.json"
                ).is_file():
                    row["reason"] = "missing annotation completion marker"
                    continue
                row.update(
                    audit_clip(
                        target_clip,
                        key,
                        runs,
                        observations,
                        quality,
                        source_manifest_sha256=source_clip.digest(),
                    )
                )
                row["status"] = "completed"
            except Exception as exc:
                row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        completed = [
            row for row in report["clips"].values() if row["status"] == "completed"
        ]
        for field in (
            "teacher_checkpoints",
            "teacher_settings_sha256",
            "dino_checkpoint_sha256",
            "dino_spec",
            "people_producer",
            "court_producer",
        ):
            if len({_digest(row[field]) for row in completed}) > 1:
                report["errors"].append(f"Mixed provenance: {field}")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["counts"] = {
        status: sum(row["status"] == status for row in report["clips"].values())
        for status in ("completed", "missing", "error", "excluded")
    }
    report["expected_count"] = len(report["clips"])
    report["complete"] = (
        not report["errors"]
        and not report["counts"]["missing"]
        and not report["counts"]["error"]
        and bool(report["counts"]["completed"])
    )
    report["status"] = (
        "complete"
        if report["complete"]
        else "incomplete"
        if not report["errors"] and not report["counts"]["error"]
        else "error"
    )
    report["metric_interpretation"] = (
        "Raw and refined use the SAME final SLCS positive frame weights; frame eligibility only, before training window selection. Reprojection is observable consistency, not independent 3D accuracy."
    )
    aggregate: dict[str, Any] = {}
    for row in report["clips"].values():
        if row["status"] != "completed":
            continue
        for stage, selections in row.get("stages", {}).items():
            for selection, metrics in selections.items():
                for name, metric in metrics.items():
                    if not isinstance(metric, dict) or not metric["count"]:
                        continue
                    key = f"{stage}.{selection}.{name}"
                    total = aggregate.setdefault(
                        key, {"count": 0, "sum": 0.0, "max": None}
                    )
                    total["count"] += metric["count"]
                    total["sum"] += metric["mean"] * metric["count"]
                    total["max"] = (
                        metric["max"]
                        if total["max"] is None
                        else max(total["max"], metric["max"])
                    )
    for total in aggregate.values():
        total["mean"] = total.pop("sum") / total["count"]
    report["aggregate_sample_weighted_metrics"] = aggregate
    save_json_atomic(report, output / "quality_report.json")
    rows = []

    def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(item, dict):
                flat.update(flatten(item, key))
            else:
                flat[key] = json.dumps(item) if isinstance(item, list) else item
        return flat

    for key, row in report["clips"].items():
        rows.append({"clip_id": key, **flatten(row)})
    fields = ["clip_id", "status"] + sorted(
        {key for row in rows for key in row} - {"clip_id", "status"}
    )
    with (output / "quality_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if report["status"] == "error" or (not report["complete"] and not allow_incomplete):
        raise ValueError(
            f"Dataset audit {report['status']}; see {output / 'quality_report.json'}"
        )
    return report
