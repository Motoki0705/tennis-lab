"""Pinned three-clip doubles CLI qualification; GPU execution is queue-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
C3 = REPO / "knowledge/runs/run-i1751-runtime-gpu-a5"
sys.path.insert(0, str(C3))
from health_guard import (  # noqa: E402
    capture_diagnostics,
    run_variant,
    utc,
    write,
    write_anomaly,
)

CLIPS = ("video_000/clip_000", "video_000/clip_037", "video_000/clip_027")
SETTINGS = Path(__file__).with_name("court-settings.json")
SOURCE = REPO / "data/tennis_multiview/processed/meiji_4cam/dataset"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def common_root():
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO, text=True).strip()).parent


def overrides():
    settings = json.loads(SETTINGS.read_text())
    if settings["overrides"][0] != "court_kp.mode=video_shared":
        raise ValueError("Qualification requires the pinned shared court")
    return [*settings["overrides"], "player_association.players_per_side=2"]


def arguments(root, phase, clips):
    settings = json.loads(SETTINGS.read_text())
    values = [f"paths.data_root={root}", f"paths.output_root={root}",
        f"paths.artifact_root={common_root() / settings['artifact_root_from_main']}",
        f"dataset_directory=datasets/{phase}", "overwrite=true", "continue_on_error=true",
        "clip_ids=" + json.dumps(list(clips), separators=(",", ":")),
        "pipeline_overrides=" + json.dumps(overrides(), separators=(",", ":")),
        f"hydra.run.dir={root / phase / 'hydra'}"]
    if phase == "batch":
        values += ["video_id=video_000", "pipeline_config=pipeline_x3d_dpt", "progress_directory=batch/progress"]
    return values


def prepare(root):
    from src.tennis_scene.pipeline.clip_inputs import prepare_video_inputs

    source = SOURCE.resolve()
    raw = (source / "dataset.json").read_bytes()
    dataset_id, clips = prepare_video_inputs(source, "video_000", CLIPS)
    if dataset_id != "meiji_4cam" or sum(c.media.source.num_frames for c in clips) != 883:
        raise ValueError("The parent-pinned three-clip timeline changed")
    destination = root / "input"
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    for clip in clips:
        for path, digest in clip.inputs.snapshot.files:
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha(target) != digest or sha(path) != digest:
                raise ValueError(f"Input changed while freezing: {path}")
            files.append({"path": str(relative), "sha256": digest, "bytes": target.stat().st_size})
        clip.inputs.snapshot.assert_unchanged()
    document = json.loads(raw)
    document["clips"] = [clip.inputs.record.to_dict() for clip in clips]
    write(destination / "dataset.json", document)
    (root / "source-dataset.json").write_bytes(raw)
    write(root / "input-manifest.json", {"dataset_id": dataset_id, "video_id": "video_000",
        "clip_order": list(CLIPS), "frames_per_camera": 883, "files": files,
        "selected_records": document["clips"], "source_dataset_sha256": hashlib.sha256(raw).hexdigest()})
    for phase in ("batch", "single"):
        dataset = root / "datasets" / phase
        shutil.copytree(destination, dataset, ignore=lambda _, names: [n for n in names if n.endswith(".mp4")])
        for media in destination.rglob("*.mp4"):
            (dataset / media.relative_to(destination)).symlink_to(media)
    verify(root)
    print(json.dumps({"frozen": list(CLIPS), "files": len(files), "frames": 883}), flush=True)


def verify(root):
    manifest = json.loads((root / "input-manifest.json").read_text())
    if tuple(manifest["clip_order"]) != CLIPS or manifest["frames_per_camera"] != 883:
        raise ValueError("Unexpected qualification inputs")
    for item in manifest["files"]:
        path = root / "input" / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
    return manifest


def worker(root, phase, clips):
    # C3's proven observer registers all-thread faulthandler before heavy imports.
    import benchmark as observer_module
    import numpy as np
    import torch

    from src.tennis_scene.generate_dataset import pseudo_annotation
    from src.tennis_scene.pipeline import video_batch
    from src.tennis_scene.pipeline.artifacts import json_value
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
    from src.tennis_scene.pipeline.runner import PlanExecutor
    from src.utils.inference_recording import (
        InferenceRecorder,
        model_event,
        recording_context,
    )

    directory = root / phase
    directory.mkdir(exist_ok=True)
    observer = observer_module.Observer(directory)
    observer.phase = "cli_" + phase
    observer.heartbeat()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = {"phase": phase, "status": "running", "started_at_utc": utc(), "clips": list(clips),
        "pipeline_config": "pipeline_x3d_dpt", "pipeline_overrides": overrides(),
        "single_preset_adapter": phase == "single", "cuda_recording_devices": ["cuda:0"]}

    @contextmanager
    def lifecycle(action, model, **attributes):
        with model_event(action, model, **attributes), observer.model_event(action, model, **attributes):
            yield

    def construct(runtime):
        if runtime.player_association.players_per_side != 2:
            raise ValueError("Doubles override was not applied by the CLI")
        required = ("person_observations", "player_reconstruction", "ball_reconstruction", "gvhmr")
        if not all(runtime.enabled[name] for name in required):
            raise ValueError("Qualification requires the complete body/ball pipeline")
        state.update(players_per_side=runtime.player_association.players_per_side,
                     enabled=dict(runtime.enabled), processing_settings=json_value(runtime.processing_settings),
                     ball_detection=json_value(runtime.ball_detection))
        return TennisSceneOrchestrator(runtime, model_event=lifecycle)

    original_batch, original_publish = video_batch.run_video_batch, video_batch.publish_annotation
    original_finalize = PlanExecutor.finalize_clip

    def measured_batch(*args, **kwargs):
        result = original_batch(*args, **kwargs, recording_devices=("cuda:0",))
        state.update(batch_returned=True, progress=str(result.progress_path),
                     recording=str(result.progress_path.with_name("recording.json")))
        return result

    def publication(inputs, *args, **kwargs):
        with recording_context(clip_id=inputs.record.clip_id, node="clip_publication"):
            return original_publish(inputs, *args, **kwargs)

    def finalize(executor, clip_id):
        with recording_context(clip_id=clip_id, node="scene_export"):
            return original_finalize(executor, clip_id)

    started = time.perf_counter()
    try:
        sys.setprofile(observer.profile)
        with (patch.object(TennisSceneOrchestrator, "from_runtime_config", side_effect=construct),
              patch.object(video_batch, "run_video_batch", side_effect=measured_batch),
              patch.object(video_batch, "publish_annotation", side_effect=publication),
              patch.object(pseudo_annotation, "publish_annotation", side_effect=publication),
              patch.object(PlanExecutor, "finalize_clip", finalize)):
            sys.argv = [f"tennis_scene.{phase}", *arguments(root, phase, clips)]
            if phase == "batch":
                from src.tennis_scene.scripts.run_video_batch import main
                main()
            else:
                import src.tennis_scene.scripts.generate_dataset as single
                compose = single._compose_pipeline_config
                with patch.object(single, "_compose_pipeline_config", side_effect=lambda cfg: compose(cfg, config_name="pipeline_x3d_dpt")):
                    recording = directory / "recording.json"
                    state["recording"] = str(recording)
                    with InferenceRecorder(recording, devices=("cuda:0",), metadata={"phase": phase, "clips": list(clips)}):
                        single.main()
        state["status"] = "complete"
        return 0
    except BaseException as error:
        sys.setprofile(None)
        detail = traceback.format_exc()
        state.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=detail)
        if observer_module.cuda_failure(detail):
            observer.cuda_error_seen = True
            write_anomaly(root, {"kind": "cuda_error", "phase": phase, "detail": detail, "pid": os.getpid()})
        if observer_module.STACK_HANDLE is not None:
            observer_module.faulthandler.dump_traceback(file=observer_module.STACK_HANDLE, all_threads=True)
        raise
    finally:
        sys.setprofile(None)
        state.update(finished_at_utc=utc(), entry_wall_seconds=time.perf_counter() - started,
                     backend_loads=observer.calls, runtime_events=observer.runtime_events)
        write(directory / "entry.json", state)
        observer_module.faulthandler.cancel_dump_traceback_later()


def reference_clip(root):
    entry = root / "batch/entry.json"
    if not entry.exists():
        return None
    state = json.loads(entry.read_text())
    if not state.get("batch_returned"):
        return None
    progress = json.loads(Path(state["progress"]).read_text())
    return next((clip for clip in CLIPS if progress["clips"][clip]["status"] == "generated"), None)


def accepted(result):
    return all(result.get(key, False) for key in (
        "all_three_completed", "doubles_identity_counts_ok", "body_exports_present",
        "runtime_lifetimes_ok", "comparison_strict")) and not result.get("cuda_anomaly", True)


def run(root, expected):
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("GPU qualification must run through the shared training queue")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != expected:
        raise ValueError(f"Expected committed measurement HEAD {expected}, got {actual}")
    paths = ["src", str(Path(__file__).parent.relative_to(REPO)), str(C3.relative_to(REPO))]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=REPO, check=True)
    verify(root)
    if (root / "terminal.json").exists():
        raise FileExistsError("One run per output directory; no automatic retry")
    terminal = {"queue_job": os.environ["TENNIS_RUN_ID"], "commit": actual,
                "started_at_utc": utc(), "status": "running", "outcomes": {},
                "players_per_side": 2, "profile": "pipeline_x3d_dpt", "clips": list(CLIPS)}
    try:
        phases = [("batch", CLIPS)]
        for phase, clips in phases:
            print(json.dumps({"starting": phase, "clips": list(clips)}), flush=True)
            result = run_variant(root, phase,
                [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "worker", "--root", str(root),
                 "--phase", phase, "--clip-ids", *clips], REPO)
            terminal["outcomes"][phase] = result
            write(root / "terminal.json", terminal)
            if (root / "anomaly.json").exists():
                if "diagnostics" not in result:
                    # The worker already dumped its stacks. Capture one bounded
                    # GPU report even when it exited before the guard saw it.
                    class Exited:
                        def __init__(self, outcome):
                            self.pid = outcome.get("pid")
                            self.returncode = outcome.get("returncode", 1)

                        def poll(self):
                            return self.returncode
                    result["diagnostics"] = capture_diagnostics(root, phase, Exited(result))
                break
            if phase == "batch":
                reference = reference_clip(root)
                terminal["reference_clip"] = reference
                if reference is not None:
                    phases.append(("single", (reference,)))
        verify(root)
        from doubles_analysis import report
        result = report(root)
        terminal["qualification"] = result
        terminal["status"] = "complete" if accepted(result) else "failed"
    except BaseException as error:
        terminal.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        terminal["finished_at_utc"] = utc()
        write(root / "terminal.json", terminal)
    return 0 if terminal["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit")
    parser.add_argument("--phase", choices=("batch", "single"))
    parser.add_argument("--clip-ids", nargs="+")
    options = parser.parse_args()
    root = options.root.resolve()
    if not root.is_relative_to((REPO / "outputs").resolve()):
        raise ValueError("Qualification writes must stay in this worktree's outputs")
    if options.action == "prepare":
        prepare(root)
        return 0
    if options.action == "worker":
        if options.phase is None or not options.clip_ids:
            parser.error("worker requires --phase and --clip-ids")
        return worker(root, options.phase, tuple(options.clip_ids))
    if options.action == "report":
        from doubles_analysis import report
        print(json.dumps(report(root), indent=2))
        return 0
    return run(root, options.expect_commit)


if __name__ == "__main__":
    raise SystemExit(main())
