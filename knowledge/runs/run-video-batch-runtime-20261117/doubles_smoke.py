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

    nodes = {"clip0": {"ball_detection/cam0": {"seconds": 7}, "court_calibration": {"seconds": 2}},
             "clip1": {"ball_detection/cam1": {"seconds": 5}, "court_calibration": {"seconds": 3}}}
    events = [{"attributes": {"node": "ball_detection/cam0"}, "stage": "model_load", "start_ns": 0, "end_ns": 2_000_000_000}]
    estimates = stage_estimates(nodes, events, 10, 2, 100, 6)
    assert estimates["ball_detection"]["estimated_69clip_seconds"] == 102  # 2 + (12 - 2) * 10
    assert estimates["court_calibration"]["estimated_69clip_seconds"] == 15  # 5 * 3
    recording = {"total_wall_seconds": 25, "events": [
        {"attributes": {"node": "scene_export"}, "depth": 0, "start_ns": 1_000_000_000, "end_ns": 4_000_000_000},
        {"attributes": {"node": "scene_export"}, "depth": 1, "start_ns": 2_000_000_000, "end_ns": 3_000_000_000},
        {"attributes": {"node": "clip_publication"}, "depth": 0, "start_ns": 5_000_000_000, "end_ns": 7_000_000_000}]}
    overhead = overhead_estimates(recording, estimates, 2, 6)
    assert [r["measured_seconds"] for r in overhead["stages"]] == [3, 2]
    assert overhead["estimated_69clip_remaining_seconds_range"] == [3, 9]  # 25 - 17 - 5

    events = [dict(model=model, status="complete", action=action, instance=model)
              for model in RUNTIME_MODELS for action in ("load", "unload")]
    assert lifetimes(events)["each_target_loaded_once"]
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
    print("settings, stage attribution, identity diagnostics, lifetimes and strict gating: passed")


if __name__ == "__main__":
    main()
