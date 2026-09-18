"""CPU replay of production root refinement on the four completed v7 clips."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.dataset_pipeline.refinement import (
    RefinementSettings,
    check_label_coverage,
    refine_scene,
)
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    source = root / "outputs/tennis_scene/generate/meiji_rgb_v7/s42-001"
    observations = root / "outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004"
    config = OmegaConf.load(
        root / "src/tennis_scene/configs/build_slcs_dataset_base.yaml"
    )
    baseline = RefinementSettings.from_config(config.refinement)
    baseline = replace(baseline, player_root_view_support="hips")
    candidate = replace(baseline, player_root_view_support="hips_and_shoulders")
    report = {"status": "running", "clips": {}, "input_sha256": {}}
    for clip_id in (
        "video_000/clip_007",
        "video_000/clip_009",
        "video_001/clip_000",
        "video_001/clip_001",
    ):
        for path in (
            source / clip_id / "scene.npz",
            source / clip_id / "scene.metadata.json",
            source / clip_id / "refined_scene.npz",
            source / clip_id / "refined_scene.metadata.json",
            observations / clip_id / "court.npz",
        ):
            report["input_sha256"][str(path)] = dual_sha256(path)
        with np.load(observations / clip_id / "court.npz") as archive:
            homographies = archive["homographies"]
        control = load_scene_result(source / clip_id / "scene.npz")
        changed = load_scene_result(source / clip_id / "scene.npz")
        recorded = load_scene_result(source / clip_id / "refined_scene.npz")
        old_evidence = refine_scene(control, homographies, baseline)
        new_evidence = refine_scene(changed, homographies, candidate)
        check_label_coverage(new_evidence, candidate)
        fields = ("player_position", "player_kp_3d", "ball_3d", "player_yaw")
        control_equal = {
            name: bool(np.array_equal(getattr(control, name), getattr(recorded, name)))
            for name in fields
        }
        assert all(control_equal.values()), (clip_id, control_equal)
        old_quality = control.metadata["label_quality"]
        recorded_quality = recorded.metadata["label_quality"]
        assert old_quality == recorded_quality, clip_id
        new_quality = changed.metadata["label_quality"]
        for name in ("ball_weight", "ball_source"):
            assert old_quality[name] == new_quality[name], (clip_id, name)
        assert np.array_equal(changed.ball_3d, control.ball_3d), clip_id
        assert np.array_equal(changed.player_yaw, control.player_yaw), clip_id
        assert np.array_equal(changed.player_position[0], control.player_position[0]), (
            clip_id
        )
        displacement = np.linalg.norm(
            changed.player_position - control.player_position, axis=-1
        )
        old_weight = np.asarray(old_quality["player_weight"])
        new_weight = np.asarray(new_quality["player_weight"])
        row = {
            "control_arrays_exact": control_equal,
            "control_label_quality_exact": True,
            "ball_and_yaw_unchanged": True,
            "player0_positions_unchanged": True,
            "old_player_label_fraction": old_evidence["player_label_fraction"],
            "new_player_label_fraction": new_evidence["player_label_fraction"],
            "max_position_change_m": displacement.max(axis=1).tolist(),
            "position_change_over_1m_frames": (displacement > 1.0).sum(axis=1).tolist(),
            "newly_excluded_frames": ((old_weight > 0) & (new_weight == 0))
            .sum(axis=1)
            .tolist(),
            "newly_supported_frames": ((old_weight == 0) & (new_weight > 0))
            .sum(axis=1)
            .tolist(),
        }
        report["clips"][clip_id] = row
        np.savez_compressed(
            args.output_dir / (clip_id.replace("/", "_") + ".npz"),
            control_position=control.player_position,
            candidate_position=changed.player_position,
            control_weight=old_weight,
            candidate_weight=new_weight,
            candidate_source=np.asarray(new_quality["player_source"]),
        )
        (args.output_dir / "results.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    report["status"] = "passed"
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "clips": report["clips"]}, indent=2))


if __name__ == "__main__":
    main()
