"""CPU checks for experiment settings, attribution, and strict success gating."""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from doubles_acceptance import CLIPS, REPO, accepted, arguments, overrides  # noqa: E402
from doubles_analysis import (  # noqa: E402
    RUNTIME_MODELS,
    identities,
    lifetimes,
    overhead_estimates,
    stage_estimates,
)


def main():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.tennis_scene.configuration import PipelineRuntimeConfig

    with initialize_config_dir(config_dir=str(REPO / "src/tennis_scene/configs"), version_base="1.3"):
        default = compose(config_name="pipeline_x3d_dpt")
        chosen = compose(config_name="pipeline_x3d_dpt", overrides=overrides())
    assert default.player_association.players_per_side == 1
    runtime = PipelineRuntimeConfig.from_config(chosen, bind_inputs=False)
    assert runtime.player_association.players_per_side == 2
    assert runtime.ball_detection.precision == "bf16"
    assert runtime.ball_detection.num_frames == 32
    assert runtime.sampling_max_frames == 2048
    assert runtime.video_court.action == "load"
    assert all(runtime.enabled[name] for name in ("person_observations", "player_reconstruction", "gvhmr", "ball_reconstruction"))
    assert OmegaConf.to_container(default)["player_association"] == {"players_per_side": 1}
    args = arguments(Path("/tmp/qualification"), "batch", CLIPS)
    assert "pipeline_config=pipeline_x3d_dpt" in args and "video_id=video_000" in args
    assert next(a for a in args if a.startswith("pipeline_overrides=")).count("players_per_side=2") == 1

    nodes = {"a": {"ball/cam0": {"seconds": 7, "status": "executed"}, "ball/cam1": {"seconds": 3, "status": "executed"}, "court_calibration": {"seconds": 2, "status": "executed"}, "hmr2": {"seconds": 4, "status": "executed"}, "court_detection/cam0": {"seconds": 3, "status": "executed"}, "court_detection/cam1": {"seconds": 1, "status": "executed"}},
             "b": {"ball/cam0": {"seconds": 5, "status": "executed"}, "ball/cam1": {"seconds": 5, "status": "executed"}, "court_calibration": {"seconds": 3, "status": "executed"}, "hmr2": {"seconds": 6, "status": "failed"}, "late": {"seconds": None, "status": "blocked"}, "court_detection/cam0": {"seconds": 2, "status": "executed"}, "court_detection/cam1": {"seconds": 4, "status": "executed"}}}
    events = [{"attributes": {"clip_id": "a", "node": "ball/cam0"}, "stage": "model_load", "start_ns": 0, "end_ns": 2_000_000_000}]
    estimates = stage_estimates(nodes, events, {"a": 10, "b": 30}, 400, 6, 2)
    assert estimates["ball"]["estimated_69clip_seconds"] == 182  # 2 + (20 - 2) * 800/80
    assert estimates["ball"]["sample_frame_units"] == 80
    assert estimates["hmr2"]["estimated_69clip_seconds"] == 160  # 4 * 400/10; failed b excluded
    assert estimates["hmr2"]["failed_node_seconds"] == 6
    assert estimates["late"]["estimated_69clip_seconds"] is None
    assert estimates["court_calibration"]["estimated_69clip_seconds"] == 15
    assert estimates["court_detection"]["estimated_69clip_seconds"] == 30  # 10s * (6 clips*2cam)/(2 clips*2cam)
    recording = {"total_wall_seconds": 60, "events": [
        {"attributes": {"clip_id": "a", "node": "scene_export"}, "status": "ok", "depth": 0, "start_ns": 1_000_000_000, "end_ns": 4_000_000_000},
        {"attributes": {"clip_id": "a", "node": "scene_export"}, "status": "ok", "depth": 1, "start_ns": 2_000_000_000, "end_ns": 3_000_000_000},
        {"attributes": {"clip_id": "a", "node": "clip_publication"}, "status": "ok", "depth": 0, "start_ns": 5_000_000_000, "end_ns": 7_000_000_000}]}
    overhead = overhead_estimates(recording, estimates, 2, 6, {"a": 10, "b": 30}, 400)
    assert [r["measured_seconds"] for r in overhead["stages"]] == [3, 2]
    assert [r["estimated_69clip_seconds_range"] for r in overhead["stages"]] == [[18, 120], [12, 80]]  # one complete clip
    assert overhead["estimated_69clip_remaining_seconds_range"] == [10, 30]  # 60 - (20 + 5 + 4 + 6 + 10) - 5

    events = [dict(model=model, status="complete", action=action, instance=model)
              for model in RUNTIME_MODELS for action in ("load", "unload")]
    assert lifetimes(events)["each_target_loaded_once"]
    overlapping = [events[0], events[2], events[1], events[3], *events[4:]]
    assert lifetimes(overlapping)["each_target_loaded_once"]
    assert lifetimes(overlapping)["maximum_owned"] == 2
    assert not lifetimes(events[:-1])["each_target_loaded_once"]
    assert not lifetimes(events + [dict(model="extra", status="failed", action="load", instance="extra")])["each_target_loaded_once"]
    sides = identities({"selection": {"-1": {"players": [8, 2], "runner_up": None}, "1": {"players": [3, 6], "runner_up": 9}},
                        "identities": [{"identity": i, "presence_frames": 7} for i in (2, 3, 6, 8)]})
    assert [v["identity"] for v in sides["-1"]["identities"]] == [8, 2]
    assert sides["1"]["count"] == 2 and sides["1"]["runner_up"] == 9
    good = dict.fromkeys(("all_three_completed", "doubles_identity_counts_ok", "body_exports_present", "runtime_lifetimes_ok", "comparison_strict"), True)
    assert accepted({**good, "cuda_anomaly": False})
    for key in good:
        assert not accepted({**good, key: False, "cuda_anomaly": False}), key
    assert not accepted({**good, "cuda_anomaly": True})
    guard_exit_checks()
    print("settings, mixed/blocked stage attribution, identity diagnostics, overlapping lifetimes, strict gating and guard exits: passed")


def guard_exit_checks():
    import tempfile

    from doubles_acceptance import run_variant

    # The real C3 guard with a CPU-only fake NVML query: preserve genuine CUDA
    # text while preventing earlier normal CUDA model events from hiding exit1.
    cases = [
        ("import sys; print('model device: cuda'); sys.exit(1)", True),
        ("import sys,traceback; print('model device: cuda');\ntry: sys.exit(1)\nexcept BaseException: sys.stdout.flush(); traceback.print_exc(); raise", False),
        ("import sys; print('Traceback (most recent call last):\\nRuntimeError: CUDA illegal memory access'); sys.exit(1)", True),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for i, (code, alarm) in enumerate(cases):
            root = Path(directory) / str(i)
            root.mkdir()
            outcome = run_variant(root, "single", [sys.executable, "-B", "-c", code], Path.cwd(),
                                  query=lambda: (0, "cpu-fake"), interval=0.01)
            assert outcome["returncode"] == 1 and outcome["status"] == "failed"
            assert (root / "anomaly.json").exists() is alarm, (i, (root / "single.log").read_text())


if __name__ == "__main__":
    main()
