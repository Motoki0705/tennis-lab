"""Pinned C1 experiment, kept with its evidence rather than production code.

Run ``freeze`` on CPU, then ``run`` exclusively through the shared GPU queue.
All writes are below --root. The source dataset is read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
CLIPS = ["video_000/clip_035", "video_000/clip_037"]
CONDITIONS = ("cold_component_cache", "completed_court_prefix", "all_node_cache_hit")
MAIN_COMMIT = "2736a24a3"
SOURCE = REPOSITORY / "data/tennis_multiview/processed/meiji_4cam/dataset"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze(root: Path) -> None:
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
    )

    destination = root / "input"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace frozen input: {destination}")
    dataset_bytes = (SOURCE / "dataset.json").read_bytes()
    dataset = load_dataset_manifest(SOURCE.resolve())
    records = [dataset.clips[clip] for clip in CLIPS]
    files = []
    for record in records:
        clip = ClipManifest.load(SOURCE / record.path)
        if (clip.clip_id, clip.num_frames, len(clip.camera_ids)) != (record.clip_id, record.num_frames, 4):
            raise ValueError(f"Unfinished/inconsistent registration: {record.clip_id}")
        paths = [SOURCE / record.path / "clip.json", *[clip.media_path(camera) for camera in clip.camera_ids]]
        for path in paths:
            relative = path.relative_to(SOURCE)
            before = sha(path)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
            if sha(output) != before or sha(path) != before:
                raise ValueError(f"Input changed while freezing: {path}")
            info = None
            if path.suffix == ".mp4":
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", str(output)],
                                       check=True, capture_output=True, text=True)
                info = json.loads(probe.stdout)["streams"][0]
                if (int(info["nb_read_frames"]), info["width"], info["height"]) != (record.num_frames, record.width, record.height):
                    raise ValueError(f"Incomplete video: {path}: {info}")
            files.append({"path": str(relative), "sha256": before, "bytes": output.stat().st_size, "probe": info})
    # New exports may append other registrations; selected records must stay fixed.
    latest = load_dataset_manifest(SOURCE.resolve())
    if any(latest.clips[r.clip_id] != r for r in records):
        raise ValueError("Selected dataset registration changed while freezing")
    snapshot = json.loads(dataset_bytes)
    snapshot["clips"] = [record.to_dict() for record in records]
    write(destination / "dataset.json", snapshot)
    (root / "source-dataset.json").write_bytes(dataset_bytes)
    manifest = {"main_commit": subprocess.check_output(["git", "rev-parse", MAIN_COMMIT], text=True).strip(),
                "source": "data/tennis_multiview/processed/meiji_4cam/dataset", "clip_order": CLIPS,
                "source_dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "selected_records": [r.to_dict() for r in records], "files": files,
                "repetitions": 3, "conditions": CONDITIONS,
                "prefix_definition": "Only completed court_detection and court_calibration artifacts are adopted; downstream artifacts and exports are absent. Cache preparation is outside the timed interval.",
                "all_hit_definition": "Force dataset marker publication (overwrite=True), keep every component artifact, require all runner nodes cached and no model load.",
                "stop_on_error": True,
                "warm_definition": "repeat 1 includes initial imports/CUDA; repeats 2/3 reuse process, not models; OS cache not dropped"}
    write(root / "input-manifest.json", manifest)
    print(json.dumps({"manifest_sha256": sha(root / "input-manifest.json"), "files": len(files), "clips": CLIPS}), flush=True)


def verify(root: Path) -> dict:
    manifest = json.loads((root / "input-manifest.json").read_text())
    if manifest["clip_order"] != CLIPS or manifest["repetitions"] != 3:
        raise ValueError("Unexpected experiment specification")
    for item in manifest["files"]:
        if sha(root / "input" / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    return manifest


def verify_source(root: Path) -> None:
    manifest = verify(root)
    from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest
    latest = load_dataset_manifest(SOURCE.resolve())
    for record in manifest["selected_records"]:
        if latest.clips[record["clip_id"]].to_dict() != record:
            raise ValueError("Selected dataset registration changed")
    for item in manifest["files"]:
        if sha(SOURCE / item["path"]) != item["sha256"]:
            raise ValueError(f"Source input changed: {item['path']}")


def prepare_prefix(dataset: Path, backup: Path) -> None:
    """Prepare an owned partial store from verified completed Court artifacts.

    Preserve the complete cold store in backup; no immutable file is changed.
    Source video paths remain the same so ordinary execution identities apply.
    """
    from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest
    from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
    for record in load_dataset_manifest(dataset).clips.values():
        root = dataset / record.path / "annotations/tennis_scene"
        index = json.loads((root / "scene.json").read_text())
        retained = {name: ref for name, ref in index["artifacts"].items()
                    if name.split("/")[0] in {"court_detection", "court_calibration"}}
        if len(retained) != 8 or not index["exports"].get("scene"):
            raise ValueError("Cold run must complete all 4 Court views before prefix preparation")
        store = ClipStore(root, index["source"])
        for ref in retained.values():
            store.descriptor(ArtifactRef(**ref))
        destination = backup / record.clip_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(destination))
        root.mkdir(parents=True)
        for ref in retained.values():
            relative = Path(ref["path"]).parent
            shutil.copytree(destination / relative, root / relative)
        index.update(artifacts=retained, exports={}, revision=index["revision"] + 1)
        write(root / "scene.json", index)
        write(backup / f"{record.clip_name}-prefix-index.json", index)


def run(root: Path, expected_commit: str, limit_gib: float) -> None:
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise RuntimeError("GPU measurements must be submitted through the shared training queue")
    # Recorder starts before heavyweight pipeline imports, to include first use.
    from src.utils.inference_recording import InferenceRecorder, span

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != expected_commit:
        raise ValueError(f"Expected instrumented commit {expected_commit}, got {head}")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "src", "knowledge/runs/run-video-batch-baseline-035037-20261116/benchmark.py"]):
        raise ValueError("Measured source/driver must be committed")
    manifest = verify(root)
    verify_source(root)
    for repeat in range(1, 4):
        dataset = root / "datasets" / f"repeat-{repeat}"
        shutil.copytree(root / "input", dataset)
        for condition in CONDITIONS:
            if condition == "completed_court_prefix":
                prepare_prefix(dataset, root / "cache-preparation" / f"repeat-{repeat}")
            output = root / "recordings" / f"repeat-{repeat}-{condition}.json"
            metadata = {"main_commit": manifest["main_commit"], "instrumented_commit": head,
                        "input_manifest_sha256": sha(root / "input-manifest.json"), "repeat": repeat,
                        "condition": condition, "clip_order": CLIPS, "process_warm": repeat > 1 or condition != "cold_component_cache",
                        "allocator_limit_gib": limit_gib}
            with InferenceRecorder(output, metadata=metadata, devices=("cuda:0",)):
                import torch
                from hydra import compose, initialize_config_dir
                from omegaconf import OmegaConf

                from src.tennis_scene.configuration import PipelineRuntimeConfig
                from src.tennis_scene.generate_dataset.manifest import ClipManifest
                from src.tennis_scene.generate_dataset.pseudo_annotation import (
                    ANNOTATION_RELATIVE_DIR,
                    generate_pseudo_annotations,
                )
                from src.tennis_scene.pipeline import TennisSceneOrchestrator
                from src.utils.configuration import PathRole

                torch.cuda.set_per_process_memory_fraction(limit_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
                with initialize_config_dir(config_dir=str(REPOSITORY / "src/tennis_scene/configs"), version_base="1.3"):
                    cfg = compose(config_name="pipeline", overrides=[f"paths.data_root={dataset}"])
                    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
                    orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime)
                    receipts = []

                    def clip_runner(video_paths, camera_ids, clip_directory, *, orchestrator=orchestrator, receipts=receipts, repeat=repeat, condition=condition):
                        try:
                            return orchestrator.run(video_paths=video_paths, video_role=PathRole.DATA, camera_ids=camera_ids,
                                                    store_root=clip_directory / ANNOTATION_RELATIVE_DIR,
                                                    clip_id=ClipManifest.load(clip_directory).clip_id)
                        finally:
                            receipts.append(dict(orchestrator.last_receipt))
                            write(root / "receipts" / f"repeat-{repeat}-{condition}.json", receipts)

                    pipeline_yaml = OmegaConf.to_yaml(cfg, resolve=True)
                    with span("measurement", operation="publication_identity"):
                        identity = orchestrator.publication_identity()
                    outcomes = generate_pseudo_annotations(dataset.resolve(), clip_runner, pipeline_config_yaml=pipeline_yaml,
                                                          publication_identity=identity, clip_ids=CLIPS,
                                                          overwrite=True, continue_on_error=False)
                    write(root / "outcomes" / f"repeat-{repeat}-{condition}.json",
                          [{"clip_id": o.clip_id, "status": o.status, "error": o.error} for o in outcomes])
                    print(json.dumps({**metadata, "outcomes": [{"clip": o.clip_id, "status": o.status, "error": o.error} for o in outcomes]}), flush=True)
            summarize(root)
            if condition == "all_node_cache_hit":
                record = json.loads(output.read_text())
                if record["models"] or any(r["status"] == "failed" or not r["stage_status"] or any(v != "cached" for v in r["stage_status"].values()) for r in receipts):
                    raise ValueError("All-node cache condition was not established")
    verify(root)
    verify_source(root)


def summarize(root: Path) -> None:
    rows = []
    for path in sorted((root / "recordings").glob("*.json")):
        record = json.loads(path.read_text())
        rows.append({k: record[k] for k in ("metadata", "status", "total_wall_seconds", "stage_wall_seconds", "stage_inclusive_seconds", "models", "batch_shapes", "resources")})
    write(root / "summary.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["freeze", "verify", "run", "summarize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", default="")
    parser.add_argument("--allocator-limit-gib", type=float, default=10.)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to((REPOSITORY / "outputs").resolve()):
        raise ValueError("All experiment output must be below outputs/")
    if args.action == "freeze":
        freeze(root)
    elif args.action == "run":
        try:
            run(root, args.expect_commit, args.allocator_limit_gib)
        except Exception as error:
            summarize(root)
            write(root / "stopped.json", {"status": "failed", "error_type": type(error).__name__,
                                         "error": str(error), "continued": False})
            verify_source(root)
            raise
    elif args.action == "verify":
        print(json.dumps(verify(root), ensure_ascii=False))
    else:
        summarize(root)


if __name__ == "__main__":
    main()
