"""Explicit CPU migration for the 56 Meiji clips from Court v8 to v9.

DINO consumes unchanged RGB; ViTPose consumes RGB and selected/interpolated boxes.
Court H does not otherwise enter either model. We rerun production selection with
both H values, authenticate the old selection, and reuse pose bytes only when all
six selection/provenance arrays agree exactly (including dtype). No model fallback.
This one-run driver creates new receipts and never edits the original receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
from src.tennis_scene.dataset_pipeline.people import (
    _people_cache_settings,
    pose_support_mask,
    select_court_halves,
)
from src.tennis_scene.dataset_pipeline.person_association import association_settings
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.utils.checksum import dual_sha256
from src.utils.schema.court import CourtConfig, court_keypoints_3d

MAIN = Path("/home/kamimura/projects/tennis-lab")
REPO = Path(__file__).resolve().parents[3]
SELECTION_KEYS = (
    "boxes",
    "track_ids",
    "observed_masks",
    "pose_supported_mask",
    "source_detection_ids",
    "detection_frame_indices",
)
POLICY = "temporal_continuity_v2; largest seed; fixed IoU and incumbent-diagonal center gates; bounded median velocity prediction; no gap reset"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


class Inputs:
    def __init__(self) -> None:
        self.before: dict[str, str] = {}

    def digest(self, path: Path) -> str:
        key = str(path.resolve())
        if key not in self.before:
            self.before[key] = dual_sha256(path)
        return self.before[key]

    def json(self, path: Path) -> dict[str, Any]:
        self.digest(path)
        value = json.loads(path.read_text())
        require(isinstance(value, dict), f"Expected object: {path}")
        return cast(dict[str, Any], value)

    def arrays(self, path: Path) -> dict[str, np.ndarray]:
        self.digest(path)
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}

    def after(self) -> dict[str, str]:
        return {path: dual_sha256(Path(path)) for path in self.before}


def exact_differences(
    old: dict[str, np.ndarray], new: dict[str, np.ndarray]
) -> dict[str, Any]:
    differences = {}
    for key in SELECTION_KEYS:
        a, b = old[key], new[key]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            differences[key] = {
                "old_dtype": str(a.dtype),
                "new_dtype": str(b.dtype),
                "old_shape": list(a.shape),
                "new_shape": list(b.shape),
                "different_elements": int(np.count_nonzero(a != b))
                if a.shape == b.shape
                else None,
            }
    return differences


def selection(
    raw: dict[str, np.ndarray],
    h: np.ndarray,
    *,
    total: int,
    fps: float,
    settings: dict[str, Any],
    camera: str,
) -> dict[str, np.ndarray]:
    indices, offsets, boxes = raw["frame_indices"], raw["offsets"], raw["boxes"]
    history: list[list[dict[str, Any]]] = [[] for _ in range(total)]
    for sample, frame_index in enumerate(indices):
        history[frame_index] = [
            {"id": int(index), "bbx_xyxy": boxes[index]}
            for index in range(offsets[sample], offsets[sample + 1])
        ]
    gap = int(float(settings["max_gap_seconds"]) * fps)
    tracks, source_ids = select_court_halves(
        history,
        h,
        half_width=float(settings["court_half_width_m"]),
        half_length=float(settings["court_half_length_m"]),
        sample_indices=indices,
        min_coverage=float(settings["min_sample_coverage"]),
        max_gap_frames=gap,
        long_gap_policy=str(settings["long_gap_policy"]),
        camera_id=camera,
        association=association_settings(settings),
    )
    observed = np.stack([tracks.observed_mask(i).numpy() for i in tracks.track_ids])
    return {
        "boxes": np.stack([tracks.tracks[i].numpy() for i in tracks.track_ids]),
        "track_ids": np.array(tracks.track_ids),
        "observed_masks": observed,
        "pose_supported_mask": np.stack(
            [pose_support_mask(mask, gap) for mask in observed]
        ),
        "source_detection_ids": source_ids,
        "detection_frame_indices": indices,
    }


def validate_raw(raw: dict[str, np.ndarray], *, total: int, stride: int) -> None:
    require(
        set(raw) == {"frame_indices", "offsets", "boxes", "scores"},
        "Raw schema mismatch",
    )
    indices, offsets, boxes, scores = (
        raw[k] for k in ("frame_indices", "offsets", "boxes", "scores")
    )
    expected = np.unique(np.r_[np.arange(0, total, stride), total - 1])
    require(
        indices.dtype == np.int64 and np.array_equal(indices, expected),
        "Raw frame indices mismatch",
    )
    require(
        offsets.dtype == np.int64 and offsets.shape == (len(indices) + 1,),
        "Raw offsets schema",
    )
    require(
        offsets[0] == 0 and (np.diff(offsets) >= 0).all() and offsets[-1] == len(boxes),
        "Raw offsets invalid",
    )
    require(
        boxes.dtype == np.float32 and boxes.shape == (len(scores), 4), "Raw box schema"
    )
    require(
        scores.dtype == np.float32 and scores.shape == (len(boxes),), "Raw score schema"
    )
    require(
        np.isfinite(boxes).all()
        and np.isfinite(scores).all()
        and (boxes[:, 2:] > boxes[:, :2]).all(),
        "Raw nonfinite/degenerate detections",
    )


def copy_new(source: Path, target: Path, expected: str) -> None:
    """Exclusive independent copy, verified before publishing a paired receipt."""
    require(dual_sha256(source) == expected, f"Source changed before copy: {source}")
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    require(dual_sha256(target) == expected, f"Copied bytes mismatch: {target}")
    target.chmod(0o444)


def court(
    inputs: Inputs,
    root: Path,
    clip: ClipManifest,
    reference: ClipManifest,
    settings: dict[str, Any],
    pin: str,
) -> np.ndarray:
    folder = root / clip.clip_id
    saved = inputs.json(folder / "court.json")
    expected_identity = {
        "checkpoint_sha256": pin,
        "clip_sha256": inputs.digest(reference.manifest_path),
        "video_sha256": {
            cam: inputs.digest(reference.media_path(cam))
            for cam in reference.camera_ids
        },
        "ball_annotation_sha256": {
            cam: inputs.digest(
                reference.clip_dir / "outsource" / f"{cam}_annotations.json"
            )
            for cam in reference.camera_ids
        },
        "settings": settings,
    }
    require(saved.get("identity") == expected_identity, f"Court identity: {folder}")
    require(
        [d["camera_id"] for d in saved["diagnostics"]] == list(clip.camera_ids),
        f"Court cameras: {folder}",
    )
    require(
        clip.video_id == reference.video_id
        and (clip.width, clip.height, clip.camera_ids)
        == (reference.width, reference.height, reference.camera_ids),
        f"Court source layout: {folder}",
    )
    for a, b in zip(clip.cameras, reference.cameras, strict=True):
        require(
            all(a.get(k) == b.get(k) for k in ("source_path", "letterbox")),
            f"Court propagation layout: {folder}",
        )
    if clip.clip_id != reference.clip_id:
        require(
            saved.get("calibration_clip_id") == reference.clip_id
            and saved.get("target_manifest_sha256")
            == inputs.digest(clip.manifest_path),
            f"Court target manifest: {folder}",
        )
        require(
            saved.get("propagation_assumption")
            == "fixed cameras within the same source recording and letterbox",
            f"Court propagation policy: {folder}",
        )
        original = inputs.json(root / reference.clip_id / "court.json")
        require(
            saved
            == {
                **original,
                "calibration_clip_id": reference.clip_id,
                "target_manifest_sha256": inputs.digest(clip.manifest_path),
                "propagation_assumption": saved["propagation_assumption"],
            },
            f"Court propagated receipt differs: {folder}",
        )
    else:
        require(
            set(saved) == {"identity", "diagnostics"},
            f"Unexpected calibration receipt: {folder}",
        )
    arrays = inputs.arrays(folder / "court.npz")
    require(set(arrays) == {"keypoints", "homographies"}, f"Court NPZ schema: {folder}")
    h, kp = arrays["homographies"], arrays["keypoints"]
    require(
        h.dtype == np.float64
        and h.shape == (3, 3, 3)
        and np.isfinite(h).all()
        and (np.abs(np.linalg.det(h)) > 0).all(),
        f"Court H invalid: {folder}",
    )
    require(
        kp.dtype == np.float32
        and kp.shape == (3, clip.num_frames, 14, 2)
        and np.isfinite(kp).all()
        and ((kp >= 0) & (kp <= 1)).all(),
        f"Court KP invalid: {folder}",
    )
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    projected = np.stack(
        [
            cv2.perspectiveTransform(physical[None], matrix)[0]
            / np.asarray((clip.width, clip.height), np.float32)
            for matrix in h
        ]
    ).astype(np.float32)
    require(np.array_equal(kp[:, 0], projected), f"Court KP/H mismatch: {folder}")
    ref = inputs.arrays(root / reference.clip_id / "court.npz")
    require(
        np.array_equal(h, ref["homographies"])
        and np.array_equal(
            kp, np.repeat(ref["keypoints"][:, :1], clip.num_frames, axis=1)
        ),
        f"Court propagation arrays: {folder}",
    )
    return h


def run(source: Path, target: Path, output: Path) -> None:
    require(
        source != target
        and not source.is_relative_to(target)
        and not target.is_relative_to(source),
        "Source/target must be separate roots",
    )
    require(
        output.is_relative_to(MAIN / "outputs") and not output.exists(),
        "Report output must be new under main outputs",
    )
    require(
        not output.is_relative_to(source) and not output.is_relative_to(target),
        "Report must be separate from observations",
    )
    output.mkdir(parents=True)
    inputs = Inputs()
    results: list[dict[str, Any]] = []
    try:
        with initialize_config_dir(
            config_dir=str(REPO / "src/tennis_scene/configs"), version_base="1.3"
        ):
            cfg = compose(config_name="build_slcs_dataset")
        with open_dict(cfg.paths):
            for key, value in {
                "data_root": MAIN / "data",
                "output_root": MAIN / "outputs",
                "checkpoint_root": MAIN / "ckpt",
                "external_asset_root": MAIN / "third_party",
            }.items():
                cfg.paths[key] = str(value)
        runtime = DatasetBuildConfig.from_config(cfg)
        paths = ReferenceClipPaths.from_config(cfg)
        pins = dict(cfg.checkpoint_sha256)
        inputs.digest(Path(__file__))
        for path in (REPO / "src").rglob("*.py"):
            inputs.digest(path)
        for path in (REPO / "src/tennis_scene/configs").rglob("*.yaml"):
            inputs.digest(path)
        inputs.digest(runtime.source / "dataset.json")
        for role, path in [
            ("dino", paths.dino_checkpoint),
            ("vitpose", paths.vitpose_checkpoint),
        ]:
            require(
                inputs.digest(path) == pins[role], f"Checkpoint pin mismatch: {role}"
            )
        manifest = load_dataset_manifest(runtime.source)
        require(
            len(runtime.dataset_clip_ids) == 56,
            "This driver requires exactly 56 configured clips",
        )
        settings = _people_cache_settings(cfg.people)
        require(
            settings["selection_policy"] == "temporal_continuity"
            and settings["long_gap_policy"] == "mask",
            "Unexpected people policies",
        )
        new_court = cast(
            dict[str, Any], OmegaConf.to_container(cfg.court, resolve=True)
        )
        require(
            new_court.get("crop_refinement_padding_px") == 20.0,
            "Expected v9 Court refinement",
        )
        old_court = {
            k: v for k, v in new_court.items() if k != "crop_refinement_padding_px"
        }
        plans = []
        for clip_id in runtime.dataset_clip_ids:
            clip = ClipManifest.load(runtime.source / manifest.clips[clip_id].path)
            require(
                clip.camera_ids == ("cam0", "cam1", "cam2"),
                f"Unexpected cameras: {clip_id}",
            )
            inputs.digest(clip.manifest_path)
            reference = ClipManifest.load(
                runtime.source
                / manifest.clips[runtime.calibration_clips[clip.video_id]].path
            )
            old_h = court(inputs, source, clip, reference, old_court, pins["court"])
            new_h = court(inputs, target, clip, reference, new_court, pins["court"])
            for ci, camera in enumerate(clip.camera_ids):
                src, dst = source / clip_id, target / clip_id
                names = [
                    f"{camera}_{stem}{suffix}"
                    for stem in ("detections", "people")
                    for suffix in (".npz", ".metadata.json")
                ]
                names.append(f"{camera}_people.reuse.json")
                require(
                    not any((dst / name).exists() for name in names),
                    f"Target already contains observations: {dst}/{camera}",
                )
                video_sha = inputs.digest(clip.media_path(camera))
                raw_identity = {
                    "schema_version": 1,
                    "video_sha256": video_sha,
                    "checkpoint_sha256": pins["dino"],
                    "confidence": float(settings["confidence"]),
                    "short_side": int(settings["short_side"]),
                    "max_long_side": int(settings["max_long_side"]),
                    "stride": int(settings["detection_stride"]),
                    "total_frames": clip.num_frames,
                }
                require(
                    inputs.json(src / names[1]) == raw_identity,
                    f"Raw identity: {src}/{camera}",
                )
                old_identity = {
                    "schema_version": 4,
                    "video_sha256": video_sha,
                    "detector_sha256": pins["dino"],
                    "pose_sha256": pins["vitpose"],
                    "settings": settings,
                    "homography": old_h[ci].tolist(),
                    "policy": POLICY,
                }
                require(
                    inputs.json(src / names[3]) == old_identity,
                    f"People identity: {src}/{camera}",
                )
                raw = inputs.arrays(src / names[0])
                validate_raw(
                    raw, total=clip.num_frames, stride=int(settings["detection_stride"])
                )
                people = inputs.arrays(src / names[2])
                require(
                    set(people) == {*SELECTION_KEYS, "keypoints"},
                    f"People schema: {src}/{camera}",
                )
                kp = people["keypoints"]
                require(
                    kp.dtype == np.float32
                    and kp.shape == (2, clip.num_frames, 17, 3)
                    and np.isfinite(kp).all(),
                    f"People keypoints: {src}/{camera}",
                )
                old_selection = selection(
                    raw,
                    old_h[ci],
                    total=clip.num_frames,
                    fps=clip.fps,
                    settings=settings,
                    camera=camera,
                )
                require(
                    not exact_differences(people, old_selection),
                    f"Old selection not reproduced: {src}/{camera}",
                )
                require(
                    (kp[..., 2][~people["pose_supported_mask"]] == 0).all(),
                    f"Unsupported confidence: {src}/{camera}",
                )
                result: dict[str, Any] = {
                    "clip_id": clip_id,
                    "camera_id": camera,
                    "source_people_sha256": inputs.digest(src / names[2]),
                    "source_receipt_sha256": inputs.digest(src / names[3]),
                    "source_court_sha256": inputs.digest(src / "court.npz"),
                    "target_court_sha256": inputs.digest(dst / "court.npz"),
                }
                try:
                    chosen = selection(
                        raw,
                        new_h[ci],
                        total=clip.num_frames,
                        fps=clip.fps,
                        settings=settings,
                        camera=camera,
                    )
                    differences = exact_differences(people, chosen)
                    result.update(
                        status="recompute_required" if differences else "reused",
                        differences=differences,
                        compared_arrays=list(SELECTION_KEYS),
                    )
                except ValueError as error:
                    result.update(
                        status="recompute_required", selection_rejection=str(error)
                    )
                results.append(result)
                plans.append(
                    (
                        src,
                        dst,
                        names,
                        {**old_identity, "homography": new_h[ci].tolist()},
                        result,
                    )
                )
        write_new(output / "inputs_before.json", inputs.before)
        checked = inputs.after()
        write_new(output / "inputs_prepublication.json", checked)
        require(checked == inputs.before, "Inputs changed during planning")
        for src, dst, names, identity, result in plans:
            for name in names[:2]:
                copy_new(src / name, dst / name, inputs.digest(src / name))
            if result["status"] == "reused":
                copy_new(src / names[2], dst / names[2], inputs.digest(src / names[2]))
                write_new(dst / names[3], identity)
                write_new(
                    dst / names[4],
                    {
                        **result,
                        "source_root": str(source),
                        "target_root": str(target),
                        "old_selection_reproduced": True,
                        "method": "production_selection_exact_dtype_shape_value",
                        "inputs_manifest": str(output / "inputs_before.json"),
                    },
                )
        after = inputs.after()
        write_new(output / "inputs_after.json", after)
        require(after == inputs.before, "Inputs changed during publication")
        write_new(
            output / "summary.json",
            {
                "status": "complete",
                "cameras": results,
                "recompute_clip_ids": sorted(
                    {
                        r["clip_id"]
                        for r in results
                        if r["status"] == "recompute_required"
                    }
                ),
                "reused_cameras": sum(r["status"] == "reused" for r in results),
                "recompute_cameras": sum(
                    r["status"] == "recompute_required" for r in results
                ),
            },
        )
    except BaseException as error:
        if not (output / "inputs_before.json").exists():
            write_new(output / "inputs_before.json", inputs.before)
        if not (output / "inputs_after.json").exists():
            try:
                write_new(output / "inputs_after.json", inputs.after())
            except (OSError, ValueError) as audit_error:
                write_new(
                    output / "inputs_after_failure.json", {"error": repr(audit_error)}
                )
        write_new(
            output / "failure.json",
            {
                "error": repr(error),
                "cameras_planned": results,
                "inputs_seen": inputs.before,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.target.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
