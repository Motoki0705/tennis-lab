"""CPU-only v9 observation reuse with explicit live-checkpoint warnings.

Historical v1 is imported only for immutable selection/schema/copy helpers.
Its module globals and implementation are never modified. Producer receipts
remain strictly pinned; live-checkpoint observations remain actual digests.
"""

from __future__ import annotations

import argparse
import importlib.util
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline.checkpoint_warning import (
    checkpoint_warning_policy,
    declared_checkpoint_identity,
    warn_checkpoint_difference,
)
from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
from src.tennis_scene.dataset_pipeline.people import _people_cache_settings
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)

REPO = Path(__file__).resolve().parents[3]
MAIN = Path("/home/kamimura/projects/tennis-lab")
HISTORICAL_DRIVER = (
    REPO / "knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/reuse.py"
)
_spec = importlib.util.spec_from_file_location(
    "historical_observation_reuse_v1", HISTORICAL_DRIVER
)
assert _spec is not None and _spec.loader is not None
historical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(historical)

require = historical.require
write_new = historical.write_new


def verify_input_snapshot(
    before: dict[str, str],
    after: dict[str, str],
    checkpoint_paths: dict[str, str],
    *,
    phase: str,
) -> None:
    """Preserve actual observations; only explicitly scoped model paths may warn."""
    require(before.keys() == after.keys(), f"Input set changed during {phase}")
    require(
        set(checkpoint_paths.values()) <= {"dino", "vitpose"},
        "Only DINO/ViTPose input paths may use checkpoint warnings",
    )
    for path, expected in before.items():
        actual = after[path]
        if actual == expected:
            continue
        role = checkpoint_paths.get(path)
        if role is not None and warn_checkpoint_difference(
            actual, expected, role=role, path=Path(path), context=f"inputs {phase}"
        ):
            continue
        raise ValueError(f"Input changed during {phase}: {path}")


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
    inputs = historical.Inputs()
    policy = ExitStack()
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
        roles = list(cfg.checkpoint_warning_roles)
        require(
            set(roles) == {"dino", "vitpose"},
            "Expected explicit DINO/ViTPose warning policy",
        )
        warning_path = output / "checkpoint_warnings.jsonl"
        warning_path.touch(exist_ok=False)
        policy.enter_context(checkpoint_warning_policy(pins, roles, warning_path))
        checkpoint_paths = {
            str(paths.dino_checkpoint.resolve()): "dino",
            str(paths.vitpose_checkpoint.resolve()): "vitpose",
        }
        inputs.digest(Path(__file__))
        inputs.digest(HISTORICAL_DRIVER)
        for path in (REPO / "src").rglob("*.py"):
            inputs.digest(path)
        for path in (REPO / "src/tennis_scene/configs").rglob("*.yaml"):
            inputs.digest(path)
        inputs.digest(runtime.source / "dataset.json")
        for role, path in [
            ("dino", paths.dino_checkpoint),
            ("vitpose", paths.vitpose_checkpoint),
        ]:
            actual = inputs.digest(path)
            require(
                actual == pins[role]
                or warn_checkpoint_difference(
                    actual,
                    pins[role],
                    role=role,
                    path=path,
                    context="initial live checkpoint pin",
                ),
                f"Checkpoint pin mismatch: {role}",
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
            old_h = historical.court(
                inputs, source, clip, reference, old_court, pins["court"]
            )
            new_h = historical.court(
                inputs, target, clip, reference, new_court, pins["court"]
            )
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
                    "policy": historical.POLICY,
                }
                require(
                    inputs.json(src / names[3]) == old_identity,
                    f"People identity: {src}/{camera}",
                )
                raw = inputs.arrays(src / names[0])
                historical.validate_raw(
                    raw, total=clip.num_frames, stride=int(settings["detection_stride"])
                )
                people = inputs.arrays(src / names[2])
                require(
                    set(people) == {*historical.SELECTION_KEYS, "keypoints"},
                    f"People schema: {src}/{camera}",
                )
                kp = people["keypoints"]
                require(
                    kp.dtype == np.float32
                    and kp.shape == (2, clip.num_frames, 17, 3)
                    and np.isfinite(kp).all(),
                    f"People keypoints: {src}/{camera}",
                )
                old_selection = historical.selection(
                    raw,
                    old_h[ci],
                    total=clip.num_frames,
                    fps=clip.fps,
                    settings=settings,
                    camera=camera,
                )
                require(
                    not historical.exact_differences(people, old_selection),
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
                    chosen = historical.selection(
                        raw,
                        new_h[ci],
                        total=clip.num_frames,
                        fps=clip.fps,
                        settings=settings,
                        camera=camera,
                    )
                    differences = historical.exact_differences(people, chosen)
                    result.update(
                        status="recompute_required" if differences else "reused",
                        differences=differences,
                        compared_arrays=list(historical.SELECTION_KEYS),
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
        verify_input_snapshot(
            inputs.before, checked, checkpoint_paths, phase="prepublication"
        )
        for src, dst, names, identity, result in plans:
            for name in names[:2]:
                historical.copy_new(src / name, dst / name, inputs.digest(src / name))
            if result["status"] == "reused":
                historical.copy_new(
                    src / names[2], dst / names[2], inputs.digest(src / names[2])
                )
                write_new(dst / names[3], identity)
                write_new(
                    dst / names[4],
                    {
                        **result,
                        "source_root": str(source),
                        "target_root": str(target),
                        "old_selection_reproduced": True,
                        "method": "production_selection_exact_dtype_shape_value",
                        "declared_checkpoint_sha256": declared_checkpoint_identity(),
                        "checkpoint_warning_evidence": str(warning_path),
                        "inputs_manifest": str(output / "inputs_before.json"),
                    },
                )
        after = inputs.after()
        write_new(output / "inputs_after.json", after)
        verify_input_snapshot(
            inputs.before, after, checkpoint_paths, phase="postpublication"
        )
        write_new(
            output / "summary.json",
            {
                "status": "complete",
                "checkpoint_warning_evidence": str(warning_path),
                "declared_checkpoint_sha256": declared_checkpoint_identity(),
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
    finally:
        policy.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.target.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
