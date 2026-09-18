"""Read-only CPU completion audit for the explicit Meiji RGB v8 dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.tasks.slcs.data.dino_tokens import (  # noqa: E402
    DinoTokenSpec,
    dino_dir,
    sample_frame_indices,
)
from src.tennis_scene.dataset_pipeline.features import (  # noqa: E402
    validated_feature_cache,
)
from src.tennis_scene.generate_dataset.manifest import (  # noqa: E402
    ClipManifest,
    load_dataset_manifest,
)
from src.utils.checksum import dual_sha256  # noqa: E402

PIN = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
NEW_CLIPS = {"video_001/clip_007", "video_002/clip_014", "video_002/clip_016"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    before: dict[str, str] = report["input_sha256_before"]

    def snapshot(path: Path) -> str:
        # Preserve lexical source paths, including separate original/v8 media paths.
        key = str(path.absolute())
        if key not in before:
            before[key] = dual_sha256(path)
        return before[key]

    require(snapshot(args.checkpoint) == PIN, "Fixed DINOv3 checkpoint SHA mismatch")
    reuse = ROOT / "knowledge/runs/run-slcs-meiji-v8-feature-reuse-v1/reuse_features.py"
    sources = [
        Path(__file__),
        reuse,
        *(
            ROOT / name
            for name in (
                "src/tasks/slcs/data/dino_tokens.py",
                "src/tasks/slcs/data/annotation.py",
                "src/tennis_scene/dataset_pipeline/features.py",
                "src/tennis_scene/generate_dataset/manifest.py",
                "src/utils/checksum.py",
                "src/utils/io.py",
                "src/tasks/slcs/configs/precompute_dino_tokens.yaml",
                "src/tasks/slcs/configs/data/default.yaml",
                "src/tasks/slcs/configs/paths/default.yaml",
            )
        ),
    ]
    for source in sources:
        snapshot(source)
    module_spec = importlib.util.spec_from_file_location(
        "meiji_feature_inventory", reuse
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(str(reuse))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(ROOT / "src/tasks/slcs/configs")
    ):
        config = compose(config_name="precompute_dino_tokens")
    raw_spec = OmegaConf.to_container(config.data.dino, resolve=True)
    if not isinstance(raw_spec, dict):
        raise TypeError("DINO spec must be a mapping")
    spec = DinoTokenSpec.from_dict({str(key): value for key, value in raw_spec.items()})
    report["spec"] = asdict(spec)
    for dataset in (args.original, args.target):
        snapshot(dataset / "dataset.json")
    original_index, target_index = (
        load_dataset_manifest(args.original),
        load_dataset_manifest(args.target),
    )
    selected = set(original_index.clips) - {"video_002/clip_001"}
    require(
        len(original_index.clips) == 57 and len(selected) == 56,
        "Expected 57 original / 56 selected clips",
    )
    require(
        set(target_index.clips) == selected,
        "v8 clip inventory differs from original minus excluded clip",
    )
    require(selected >= NEW_CLIPS, "Missing newly generated clips")
    # Preflight EVERY marker before any validation. No migration/write API is called.
    for clip_id in sorted(selected):
        marker = (
            dino_dir(args.target / target_index.clips[clip_id].path) / "annotation.json"
        )
        require(marker.is_file(), f"Missing completion marker: {marker}")
    for clip_id in sorted(selected):
        original_dir = args.original / original_index.clips[clip_id].path
        target_dir = args.target / target_index.clips[clip_id].path
        for directory in (original_dir, target_dir):
            snapshot(directory / "clip.json")
        original, target = (
            ClipManifest.load(original_dir),
            ClipManifest.load(target_dir),
        )
        require(
            original_index.clips[clip_id] == target_index.clips[clip_id],
            f"Root clip record mismatch: {clip_id}",
        )
        require(
            original.digest() == target.digest()
            and original.camera_ids == target.camera_ids
            and original.video_paths == target.video_paths
            and original.num_frames == target.num_frames,
            f"Clip manifest mismatch: {clip_id}",
        )
        require(len(target.camera_ids) == 3, f"Expected three cameras: {clip_id}")
        media = {}
        for camera in target.camera_ids:
            media[camera] = snapshot(original.media_path(camera))
            require(
                snapshot(target.media_path(camera)) == media[camera],
                f"Original/v8 video mismatch: {clip_id}/{camera}",
            )
        directory = dino_dir(target_dir)
        files = module._inventory(directory, target.camera_ids)
        for path in files:
            snapshot(path)
        identity: dict[str, object] = {
            "script": "src/tennis_scene/scripts/build_slcs_dataset.py",
            "checkpoint_sha256": PIN,
            "video_sha256": media,
        }
        require(
            validated_feature_cache(target, spec, identity),
            f"Missing feature completion: {clip_id}",
        )
        marker_data = json.loads((directory / "annotation.json").read_text())
        expected_frames = sample_frame_indices(target.num_frames, spec.frame_stride)
        cameras = {}
        for camera in target.camera_ids:
            with np.load(directory / f"{camera}.npz", allow_pickle=False) as saved:
                require(
                    set(saved.files) == {"tokens", "frame_idx"},
                    f"Unexpected NPZ keys: {clip_id}/{camera}",
                )
                tokens, frames = saved["tokens"], saved["frame_idx"]
                require(
                    tokens.dtype == np.float16 and frames.dtype == np.int64,
                    f"NPZ dtype mismatch: {clip_id}/{camera}",
                )
                require(
                    np.array_equal(frames, expected_frames),
                    f"Exact sampling mismatch: {clip_id}/{camera}",
                )
                require(
                    marker_data["cameras"][camera]["num_samples"] == len(frames),
                    f"Marker sample count mismatch: {clip_id}/{camera}",
                )
                cameras[camera] = {
                    "tokens_shape": list(tokens.shape),
                    "tokens_dtype": str(tokens.dtype),
                    "frame_idx_dtype": str(frames.dtype),
                    "num_samples": len(frames),
                    "first_frame": int(frames[0]),
                    "last_frame": int(frames[-1]),
                }
        report["clips"][clip_id] = {
            "status": "passed",
            "newly_generated": clip_id in NEW_CLIPS,
            "num_frames": target.num_frames,
            "manifest_digest": target.digest(),
            "video_sha256": media,
            "cameras": cameras,
        }
        print(f"{len(report['clips'])}/56 {clip_id}: passed", flush=True)
    report["counts"] = {
        "clips": len(report["clips"]),
        "cameras": sum(len(row["cameras"]) for row in report["clips"].values()),
        "new_clips": len(NEW_CLIPS),
        "reused_clips": len(selected - NEW_CLIPS),
        "clip_frames": sum(row["num_frames"] for row in report["clips"].values()),
        "camera_token_samples": sum(
            cam["num_samples"]
            for row in report["clips"].values()
            for cam in row["cameras"].values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Requires every existing marker; never invokes cache generation or migration. Existing audit output is refused.",
    )
    for name in ("original", "target", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.original, args.target = args.original.absolute(), args.target.absolute()
    require(
        args.target.name == "meiji_rgb_v8", "This audit is restricted to explicit v8"
    )
    require(
        not args.output.resolve().is_relative_to(args.original.resolve())
        and not args.output.resolve().is_relative_to(args.target.resolve()),
        "Output cannot be inside either dataset",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "failed",
        "source": {name: str(value) for name, value in vars(args).items()},
        "checkpoint_sha256": PIN,
        "clips": {},
        "input_sha256_before": {},
        "input_sha256_after": {},
        "errors": [],
    }
    started = time.monotonic()
    with args.output.open("x") as output:
        try:
            run(args, report)
        except Exception as exc:
            report["errors"].append(
                {"stage": "validation", "type": type(exc).__name__, "error": str(exc)}
            )
        finally:
            print("Checking all input hashes again", flush=True)
            for name, before in report["input_sha256_before"].items():
                try:
                    after = dual_sha256(Path(name))
                    report["input_sha256_after"][name] = after
                    require(after == before, f"Input changed: {name}")
                except Exception as exc:
                    report["errors"].append(
                        {
                            "stage": "post_hash",
                            "source": name,
                            "type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            report["status"] = "passed" if not report["errors"] else "failed"
            report["elapsed_seconds"] = time.monotonic() - started
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in ("status", "counts", "elapsed_seconds", "errors")
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
