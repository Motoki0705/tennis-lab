"""Strict CPU-only v8 -> v9 RGB reuse; materializes only original manifests/media.

Requires a NEW output directory. All 56 source caches are validated before any
feature publication. Prior audit failures remain evidence, even when this run passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
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
SPEC = DinoTokenSpec("dinov3_vitb16", 16, 256, 448, 768, 10)
LEGACY = ROOT / "knowledge/runs/run-slcs-meiji-v8-feature-reuse-v1/reuse_features.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def legacy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verified_feature_reuse_v8", LEGACY)
    if spec is None or spec.loader is None:
        raise ImportError(str(LEGACY))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HashAudit:
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.before: dict[str, str] = report.setdefault("input_sha256_before", {})
        self.after: dict[str, str] = report.setdefault("input_sha256_after", {})

    def capture(self, path: Path) -> str:
        key = str(path.absolute())
        if key not in self.before:
            self.before[key] = dual_sha256(path)
        return self.before[key]

    def finish(self, stage: str = "post_hash") -> None:
        destination = (
            self.after
            if stage == "post_hash"
            else self.report.setdefault("input_sha256_prepublication", {})
        )
        for name, expected in self.before.items():
            try:
                actual = dual_sha256(Path(name))
                destination[name] = actual
                require(actual == expected, f"Changed input bytes: {name}")
            except Exception as exc:
                self.report["errors"].append(
                    {
                        "stage": stage,
                        "source": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )


def validate_versions(original: Path, source: Path, target: Path) -> None:
    require(
        source.name == "meiji_rgb_v8" and target.name == "meiji_rgb_v9",
        "Expected source v8 and target v9",
    )
    require(
        original.parts[-3:] == ("processed", "meiji_3cam", "dataset"),
        "Expected original Meiji dataset",
    )
    require(
        len({original.resolve(), source.resolve(), target.resolve()}) == 3,
        "Dataset roots must differ",
    )
    for path in (original, source, target):
        require(
            path.is_absolute()
            and path.resolve() == path
            and (path.is_dir() or (path == target and not path.exists())),
            f"Root missing or aliased: {path}",
        )


def validate_cache(
    clip: ClipManifest,
    spec: DinoTokenSpec,
    identity: dict[str, object],
    hashes: HashAudit,
    legacy: ModuleType,
) -> dict[str, str]:
    directory = dino_dir(clip.clip_dir)
    files = legacy._inventory(directory, clip.camera_ids)
    expected = {path.name: hashes.capture(path) for path in files}
    require(
        validated_feature_cache(clip, spec, identity), f"Missing marker: {directory}"
    )
    marker = json.loads((directory / "annotation.json").read_text())
    frames_expected = sample_frame_indices(clip.num_frames, spec.frame_stride)
    for camera in clip.camera_ids:
        with np.load(directory / f"{camera}.npz", allow_pickle=False) as saved:
            require(set(saved.files) == {"tokens", "frame_idx"}, "Unexpected NPZ keys")
            tokens, frames = saved["tokens"], saved["frame_idx"]
            require(
                tokens.dtype == np.float16 and frames.dtype == np.int64,
                "Invalid raw NPZ dtypes",
            )
            require(
                np.array_equal(frames, frames_expected), "Exact frame sampling mismatch"
            )
            require(
                marker["cameras"][camera]["num_samples"] == len(frames),
                "Marker sample count mismatch",
            )
    return expected


def prepare_clip(
    original: ClipManifest,
    source: ClipManifest,
    target: ClipManifest,
    spec: DinoTokenSpec,
    hashes: HashAudit,
    legacy: ModuleType,
) -> tuple[dict[str, object], dict[str, str]]:
    for clip in (original, source, target):
        hashes.capture(clip.clip_dir / "clip.json")
        require(
            clip.digest() == original.digest()
            and clip.camera_ids == original.camera_ids
            and clip.num_frames == original.num_frames
            and clip.video_paths == original.video_paths,
            f"Manifest identity mismatch: {clip.clip_dir}",
        )
    media = {}
    for camera in original.camera_ids:
        expected = hashes.capture(original.media_path(camera))
        media[camera] = expected
        for clip in (source, target):
            require(
                hashes.capture(clip.media_path(camera)) == expected,
                f"Video identity mismatch: {clip.clip_dir}/{camera}",
            )
    identity: dict[str, object] = {
        "script": "src/tennis_scene/scripts/build_slcs_dataset.py",
        "checkpoint_sha256": PIN,
        "video_sha256": media,
    }
    source_hashes = validate_cache(source, spec, identity, hashes, legacy)
    if dino_dir(target.clip_dir).exists() or dino_dir(target.clip_dir).is_symlink():
        require(
            validate_cache(target, spec, identity, hashes, legacy) == source_hashes,
            "Existing target differs from source bytes",
        )
    return identity, source_hashes


def publish_clip(
    original: ClipManifest,
    source: ClipManifest,
    target: ClipManifest,
    spec: DinoTokenSpec,
    identity: dict[str, object],
    expected: dict[str, str],
    hashes: HashAudit,
    legacy: ModuleType,
    report: dict[str, Any],
) -> None:
    row = legacy.migrate_clip(original, target, source.clip_dir, spec, PIN)
    report["clips"][target.clip_id] = row
    require(
        row["status"] in {"linked", "already_valid"},
        f"Feature reuse incomplete: {target.clip_id}: {row}",
    )
    require(
        validate_cache(target, spec, identity, hashes, legacy) == expected,
        f"Published bytes differ from source: {target.clip_id}",
    )
    print(f"{target.clip_id}: {row['status']}", flush=True)


def run(args: argparse.Namespace, report: dict[str, Any], hashes: HashAudit) -> None:
    validate_versions(args.original, args.source, args.target)
    require(hashes.capture(args.checkpoint) == PIN, "Fixed checkpoint mismatch")
    config_path = ROOT / "src/tennis_scene/configs/build_slcs_dataset.yaml"
    data_config = ROOT / "src/tasks/slcs/configs/data/default.yaml"
    sources = [
        Path(__file__),
        LEGACY,
        config_path,
        data_config,
        *(
            ROOT / name
            for name in (
                "src/tasks/slcs/configs/precompute_dino_tokens.yaml",
                "src/tasks/slcs/configs/paths/default.yaml",
                "src/tasks/slcs/data/dino_tokens.py",
                "src/tasks/slcs/data/annotation.py",
                "src/tennis_scene/dataset_pipeline/features.py",
                "src/tennis_scene/dataset_pipeline/build.py",
                "src/tennis_scene/generate_dataset/manifest.py",
                "src/utils/checksum.py",
                "src/utils/io.py",
            )
        ),
    ]
    for path in sources:
        hashes.capture(path)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(config, dict):
        raise TypeError("Invalid build configuration mapping")
    with initialize_config_dir(
        version_base="1.3", config_dir=str(ROOT / "src/tasks/slcs/configs")
    ):
        feature_config = compose(config_name="precompute_dino_tokens")
    feature_spec = OmegaConf.to_container(feature_config.data.dino, resolve=True)
    require(
        config["dataset_output_directory"] == "slcs/meiji_rgb_v9", "Config is not v9"
    )
    require(config["checkpoint_sha256"]["dinov3"] == PIN, "Config checkpoint mismatch")
    require(
        set(config["excluded_clips"]) == {"video_002/clip_001"},
        "Fixed exclusion mismatch",
    )
    require(feature_spec == asdict(SPEC), "Fixed RGB spec mismatch")
    report["spec"] = asdict(SPEC)
    for name in (
        "run-slcs-meiji-v8-features-missing-v1",
        "run-slcs-meiji-video-alias-audit-v1",
    ):
        path = ROOT / f"knowledge/runs/{name}/audit.json"
        report.setdefault("prior_evidence", {})[str(path)] = hashes.capture(path)
    report["interpretation"] = (
        "This is a new audit; the prior video SHA mismatch remains unresolved despite a passed separate-process snapshot control. No root cause is inferred."
    )
    for root in (args.original, args.source):
        hashes.capture(root / "dataset.json")
    original_index, source_index = [
        load_dataset_manifest(root) for root in (args.original, args.source)
    ]
    selected = set(original_index.clips) - {"video_002/clip_001"}
    require(
        len(original_index.clips) == 57 and len(selected) == 56,
        "Expected 57 original / 56 selected clips",
    )
    require(
        set(source_index.clips) == selected,
        "Source/target clip inventory mismatch",
    )
    from src.tennis_scene.dataset_pipeline.build import materialize_dataset

    # Authenticate original inputs before materialization reads/copies/links them.
    for clip_id in sorted(selected):
        directory = args.original / original_index.clips[clip_id].path
        hashes.capture(directory / "clip.json")
        original_clip = ClipManifest.load(directory)
        for camera in original_clip.camera_ids:
            hashes.capture(original_clip.media_path(camera))
    if (args.target / "dataset.json").exists():
        hashes.capture(args.target / "dataset.json")
    materialize_dataset(args.original, args.target, clip_ids=tuple(sorted(selected)))
    hashes.capture(args.target / "dataset.json")
    target_index = load_dataset_manifest(args.target)
    require(set(target_index.clips) == selected, "Target clip inventory mismatch")
    legacy = legacy_module()
    prepared = []
    # All inputs and all existing targets pass before the first feature hardlink is created.
    for clip_id in sorted(selected):
        record = original_index.clips[clip_id]
        require(
            source_index.clips[clip_id] == target_index.clips[clip_id] == record,
            f"Root manifest record mismatch: {clip_id}",
        )
        manifests = []
        for root in (args.original, args.source, args.target):
            directory = root / record.path
            hashes.capture(directory / "clip.json")
            manifests.append(ClipManifest.load(directory))
        original, source, target = manifests
        require(len(original.camera_ids) == 3, f"Expected 3 cameras: {clip_id}")
        identity, expected = prepare_clip(
            original, source, target, SPEC, hashes, legacy
        )
        prepared.append((original, source, target, identity, expected))
        print(f"Preflight {clip_id}", flush=True)
    # Verify bytes again before publishing; failures stop without cache repair.
    hashes.finish("prepublication_hash")
    require(not report["errors"], "Input bytes changed during preflight")
    for original, source, target, identity, expected in prepared:
        publish_clip(
            original, source, target, SPEC, identity, expected, hashes, legacy, report
        )
    require(len(report["clips"]) == 56, "Incomplete feature reuse")
    report["counts"] = {
        "clips": 56,
        "cameras": 168,
        "linked": sum(row["status"] == "linked" for row in report["clips"].values()),
        "already_valid": sum(
            row["status"] == "already_valid" for row in report["clips"].values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "source", "target", "checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("original", "source", "target", "checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).absolute())
    require(
        not any(
            args.output_dir.resolve().is_relative_to(root.resolve())
            for root in (args.original, args.source, args.target)
        ),
        "Receipt output must be outside dataset trees",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = args.output_dir / "audit.json"
    report: dict[str, Any] = {
        "status": "running",
        "sources": {key: str(value) for key, value in vars(args).items()},
        "checkpoint_sha256": PIN,
        "clips": {},
        "errors": [],
    }
    hashes = HashAudit(report)
    with receipt.open("x") as handle:
        try:
            run(args, report, hashes)
        except Exception as exc:
            report["errors"].append(
                {"stage": "reuse", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            hashes.finish()
            report["status"] = "failed" if report["errors"] else "passed"
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "errors": report["errors"],
                "receipt": str(receipt),
            }
        )
    )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
