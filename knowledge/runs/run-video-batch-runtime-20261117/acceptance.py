"""C4 paired GPU acceptance on a fixed C1 subset, with independent stores.

Freeze on CPU. Run through the shared queue. Keep original media read-only.
One cold/cache pair per entry measures lifecycle and output agreement. The
three-repeat full performance qualification remains C10's responsibility.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
C1 = REPOSITORY / "knowledge/runs/run-video-batch-baseline-20261116/benchmark.py"


def write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def operation(root: Path, **details: object) -> None:
    event = {"utc": datetime.now(UTC).isoformat(), **details}
    write(root / "current-operation.json", event)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def copy_dataset(root: Path, name: str, clips: list[str]) -> Path:
    original = root / "input"
    dataset = root / "datasets" / name
    shutil.copytree(original, dataset, ignore=lambda _, names: [n for n in names if n.endswith(".mp4")])
    # Canonical source paths are identical for both entries. Their immutable
    # component and scene identities can therefore be compared directly.
    for media in original.rglob("*.mp4"):
        (dataset / media.relative_to(original)).symlink_to(media)
    manifest = json.loads((dataset / "dataset.json").read_text())
    manifest["clips"] = [record for record in manifest["clips"] if record["clip_id"] in clips]
    write(dataset / "dataset.json", manifest)
    return dataset


def compare(first: Path, second: Path, clips: list[str]) -> dict:
    import numpy as np

    differences = []
    compared = 0
    scenes = 0
    for clip_id in clips:
        video, clip = clip_id.split("/")
        relative = Path("videos") / video / "clips" / clip / "annotations/tennis_scene"
        roots = first / relative, second / relative
        indexes = [json.loads((r / "scene.json").read_text()) if (r / "scene.json").exists() else {"artifacts": {}} for r in roots]
        left, right = (i["artifacts"] for i in indexes)
        if set(left) != set(right):
            differences.append({"clip": clip_id, "nodes_only_single": sorted(set(left) - set(right)),
                                "nodes_only_batch": sorted(set(right) - set(left))})
        for node in left.keys() & right.keys():
            descriptors = [json.loads((r / refs[node]["path"]).read_text()) for r, refs in zip(roots, (left, right), strict=True)]
            a, b = descriptors
            compared += 1
            if a["payload"] != b["payload"] or set(a["arrays"]) != set(b["arrays"]):
                differences.append({"clip": clip_id, "node": node, "field": "payload/array names"})
                continue
            for key in a["arrays"]:
                arrays = [np.load((r / refs[node]["path"]).parent / key, allow_pickle=False)
                          for r, refs in zip(roots, (left, right), strict=True)]
                x, y = arrays
                equal = (x.shape == y.shape and x.dtype == y.dtype and
                         (np.allclose(x, y, rtol=1e-5, atol=1e-5, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)))
                if not equal:
                    differences.append({"clip": clip_id, "node": node, "array": key})
        scenes += int(all((r / "annotation.json").exists() for r in roots))
    return {"compared_components": compared, "matching_completed_scenes": scenes,
            "differences": differences, "all_requested_scenes_complete": scenes == len(clips)}


def run(root: Path, clips: list[str], court_settings: Path | None) -> int:
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.tennis_scene.configuration import PipelineRuntimeConfig
    from src.tennis_scene.generate_dataset.pseudo_annotation import (
        generate_pseudo_annotations,
    )
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
    from src.tennis_scene.pipeline.video_batch import run_video_batch
    from src.utils.configuration import PathRole
    from src.utils.inference_recording import InferenceRecorder, model_event

    @contextmanager
    def lifecycle(action, model, **attributes):
        operation(root, operation=f"model_{action}", model=model, state="started")
        with model_event(action, model, **attributes):
            yield
        operation(root, operation=f"model_{action}", model=model, state="finished")

    c1 = runpy.run_path(str(C1))
    manifest = c1["verify"](root)
    if not clips or len(set(clips)) != len(clips) or set(clips) - set(manifest["clip_order"]):
        raise ValueError("Acceptance clips must be a nonempty unique subset of the pinned C1 inputs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "diff", "HEAD", "--", "src", str(Path(__file__).relative_to(REPOSITORY))]):
        raise ValueError("Commit measured source and driver before GPU execution")
    overrides = [f"paths.data_root={root}", f"paths.output_root={root / 'outputs'}", f"paths.artifact_root={root / 'outputs'}"]
    if court_settings is not None:
        settings = json.loads(court_settings.read_text())
        if set(settings) != {"artifact_root_from_main", "overrides"} or any(not v.startswith("court_kp.") for v in settings["overrides"]):
            raise ValueError("Court settings require an explicit artifact root and court-only overrides")
        common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True).strip()).parent
        overrides[2] = f"paths.artifact_root={common / settings['artifact_root_from_main']}"
        overrides.extend(settings["overrides"])
        write(root / "court-settings.json", settings)
    with initialize_config_dir(version_base="1.3", config_dir=str(REPOSITORY / "src/tennis_scene/configs")):
        cfg = compose(config_name="pipeline_x3d_dpt", overrides=overrides)
    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
    yaml = OmegaConf.to_yaml(cfg, resolve=True)
    write(root / "pipeline.yaml.json", {"yaml": yaml})
    results = {}
    datasets = {name: copy_dataset(root, name, clips) for name in ("single", "batch")}
    for name, dataset in datasets.items():
        for cache in ("cold", "cache"):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            orchestrator = TennisSceneOrchestrator(runtime, model_event=lifecycle)
            condition = f"{name}-{cache}"
            operation(root, operation="condition", condition=condition, state="started")
            entry_started = perf_counter()
            if name == "single":
                recorder = InferenceRecorder(root / "recordings" / f"{condition}.json", devices=("cuda:0",),
                    metadata={"entry": "generate_dataset", "cache": cache, "commit": head, "clip_order": clips})
                with recorder:
                    def reconstruct(paths, cameras, directory, scene_orchestrator=orchestrator, phase=condition):
                        operation(root, operation="clip", condition=phase, clip_id=f"video_000/{directory.name}", state="started")
                        scene = scene_orchestrator.run(paths, video_role=PathRole.DATA, camera_ids=cameras,
                            store_root=directory / "annotations/tennis_scene", clip_id=f"video_000/{directory.name}")
                        operation(root, operation="clip", condition=phase, clip_id=f"video_000/{directory.name}", state="finished")
                        return scene
                    outcomes = generate_pseudo_annotations(dataset, reconstruct, pipeline_config_yaml=yaml,
                        publication_identity=orchestrator.publication_identity(), overwrite=True, continue_on_error=False,
                        scene_export=orchestrator.completed_scene_export)
                    outcome = [{"clip_id": o.clip_id, "status": o.status, "error": o.error} for o in outcomes]
                recording = recorder.destination
            else:
                batch = run_video_batch(dataset, "video_000", orchestrator, pipeline_config_yaml=yaml, overwrite=True,
                    continue_on_error=False, progress_directory=root / "progress" / condition, recording_devices=("cuda:0",))
                outcome = [{"clip_id": o.clip_id, "status": o.status, "error": o.error} for o in batch.outcomes]
                recording = batch.progress_path.with_name("recording.json")
            entry_seconds = perf_counter() - entry_started
            raw = json.loads(recording.read_text())
            results[condition] = {"outcomes": outcome, "entry_wall_seconds": entry_seconds,
                                  "total_wall_seconds": raw["total_wall_seconds"],
                                  "models": raw["models"], "resources": raw["resources"],
                                  "recording": str(recording.relative_to(root))}
            write(root / "metrics.json", {"commit": head, "clip_order": clips, "conditions": results})
            print(json.dumps({"condition": condition, "seconds": raw["total_wall_seconds"], "outcomes": outcome}), flush=True)
    comparison = compare(datasets["single"], datasets["batch"], clips)
    write(root / "comparison.json", comparison)
    write(root / "metrics.json", {"commit": head, "clip_order": clips, "conditions": results, "comparison": comparison})
    return 0 if comparison["all_requested_scenes_complete"] and not comparison["differences"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--clip-ids", nargs="+", default=["video_000/clip_000", "video_000/clip_001"])
    parser.add_argument("--court-settings", type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    if options.action == "freeze":
        runpy.run_path(str(C1))["freeze"](root)
        return 0
    try:
        return run(root, options.clip_ids, options.court_settings)
    except BaseException as error:
        current = root / "current-operation.json"
        write(root / "failure.json", {
            "utc": datetime.now(UTC).isoformat(), "pid": os.getpid(), "command": sys.argv,
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "last_operation": json.loads(current.read_text()) if current.exists() else None,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
