"""C4 paired GPU acceptance on C1's frozen three clips, with independent stores.

Freeze on CPU. Run through the shared queue. Keep original media read-only.
One cold/cache pair per entry measures lifecycle and output agreement. The
three-repeat full performance qualification remains C10's responsibility.
"""

from __future__ import annotations

import argparse
import json
import random
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
C1 = REPOSITORY / "knowledge/runs/run-video-batch-baseline-20261116/benchmark.py"


def write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def copy_dataset(root: Path, name: str) -> Path:
    original = root / "input"
    dataset = root / "datasets" / name
    shutil.copytree(original, dataset, ignore=lambda _, names: [n for n in names if n.endswith(".mp4")])
    # Canonical source paths are identical for both entries. Their immutable
    # component and scene identities can therefore be compared directly.
    for media in original.rglob("*.mp4"):
        (dataset / media.relative_to(original)).symlink_to(media)
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
                          for r, refs, d in zip(roots, (left, right), descriptors, strict=True)]
                x, y = arrays
                equal = (x.shape == y.shape and x.dtype == y.dtype and
                         (np.allclose(x, y, rtol=1e-5, atol=1e-5, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)))
                if not equal:
                    differences.append({"clip": clip_id, "node": node, "array": key})
        scenes += int(all((r / "annotation.json").exists() for r in roots))
    return {"compared_components": compared, "matching_completed_scenes": scenes,
            "differences": differences, "all_requested_scenes_complete": scenes == len(clips)}


def run(root: Path) -> int:
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
    from src.utils.inference_recording import InferenceRecorder

    c1 = runpy.run_path(str(C1))
    manifest = c1["verify"](root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "diff", "HEAD", "--", "src", str(Path(__file__).relative_to(REPOSITORY))]):
        raise ValueError("Commit measured source and driver before GPU execution")
    with initialize_config_dir(version_base="1.3", config_dir=str(REPOSITORY / "src/tennis_scene/configs")):
        cfg = compose(config_name="pipeline_x3d_dpt", overrides=[f"paths.data_root={root}",
            f"paths.output_root={root / 'outputs'}", f"paths.artifact_root={root / 'outputs'}"])
    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
    yaml = OmegaConf.to_yaml(cfg, resolve=True)
    write(root / "pipeline.yaml.json", {"yaml": yaml})
    results = {}
    datasets = {name: copy_dataset(root, name) for name in ("single", "batch")}
    for name, dataset in datasets.items():
        for cache in ("cold", "cache"):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            orchestrator = TennisSceneOrchestrator(runtime)
            condition = f"{name}-{cache}"
            if name == "single":
                recorder = InferenceRecorder(root / "recordings" / f"{condition}.json", devices=("cuda:0",),
                    metadata={"entry": "generate_dataset", "cache": cache, "commit": head, "clip_order": manifest["clip_order"]})
                with recorder:
                    def reconstruct(paths, cameras, directory, scene_orchestrator=orchestrator):
                        return scene_orchestrator.run(paths, video_role=PathRole.DATA, camera_ids=cameras,
                            store_root=directory / "annotations/tennis_scene", clip_id=f"video_000/{directory.name}")
                    outcomes = generate_pseudo_annotations(dataset, reconstruct, pipeline_config_yaml=yaml,
                        publication_identity=orchestrator.publication_identity(), overwrite=True)
                    outcome = [{"clip_id": o.clip_id, "status": o.status, "error": o.error} for o in outcomes]
                recording = recorder.destination
            else:
                batch = run_video_batch(dataset, "video_000", orchestrator, pipeline_config_yaml=yaml, overwrite=True,
                    progress_directory=root / "progress" / condition, recording_devices=("cuda:0",))
                outcome = [{"clip_id": o.clip_id, "status": o.status, "error": o.error} for o in batch.outcomes]
                recording = batch.progress_path.with_name("recording.json")
            raw = json.loads(recording.read_text())
            results[condition] = {"outcomes": outcome, "total_wall_seconds": raw["total_wall_seconds"],
                                  "models": raw["models"], "resources": raw["resources"],
                                  "recording": str(recording.relative_to(root))}
            write(root / "metrics.json", {"commit": head, "conditions": results})
            print(json.dumps({"condition": condition, "seconds": raw["total_wall_seconds"], "outcomes": outcome}), flush=True)
    comparison = compare(datasets["single"], datasets["batch"], manifest["clip_order"])
    write(root / "comparison.json", comparison)
    write(root / "metrics.json", {"commit": head, "conditions": results, "comparison": comparison})
    return 0 if comparison["all_requested_scenes_complete"] and not comparison["differences"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", required=True, type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    if options.action == "freeze":
        runpy.run_path(str(C1))["freeze"](root)
        return 0
    return run(root)


if __name__ == "__main__":
    raise SystemExit(main())
