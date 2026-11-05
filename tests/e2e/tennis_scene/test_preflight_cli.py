"""The installation CLI must preserve its nonzero exit and complete diagnostic."""

import json
import os
import subprocess
import sys
from pathlib import Path

from src.utils.paths import PROJECT_ROOT


def test_empty_installation_lists_every_asset_and_exits_nonzero(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "src.tennis_scene.scripts.preflight",
        f"paths.data_root={tmp_path / 'data'}", f"paths.checkpoint_root={tmp_path / 'ckpt'}",
        f"paths.external_asset_root={tmp_path / 'third_party'}", f"paths.output_root={tmp_path / 'outputs'}"],
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                               "TENNIS_LAB_DINO_EXTENSION": str(tmp_path / "missing.so")},
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert report["inference_executed"] is False
    extension = next(check for check in report["checks"] if check["name"] == "dino_extension")
    assert extension["status"] == "error"
    assert "TENNIS_LAB_DINO_EXTENSION" in extension["detail"]
    assert str(tmp_path / "missing.so") in extension["detail"]
    failures = {check["name"] for check in report["checks"] if check["status"] != "ok"}
    assert {"court", "ball", "detector", "vitpose", "hmr2", "gvhmr", "aflink", "tracking_encoder",
            "association_encoder", "body_model", "dinov3", "smpl_faces", "dino_models", "dinov3_hub",
            "smpl_hmr2_mean_params", "smpl_smplx_to_smpl", "smpl_smpl_coco17_regressor",
            "smpl_smplx_verts437", "smpl_smpl_neutral_joint_regressor", "video_cam0", "video_cam1", "video_cam2"} <= failures


def test_unsafe_paths_corruption_and_missing_assets_share_one_json_report(tmp_path: Path) -> None:
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"outside checkpoint must remain untouched")
    checkpoint = tmp_path / "ckpt"
    court = checkpoint / "court_detection/multiscale_depth3/b863df1f01f0.ckpt"
    court.parent.mkdir(parents=True)
    court.symlink_to(outside)
    ball = checkpoint / "ball_detection/i935-mixed-ft-s42-epoch09.ckpt"
    ball.parent.mkdir(parents=True)
    ball.write_bytes(b"corrupt ball checkpoint")
    encoder = checkpoint / "player_association/person_vit_clip_reid.pth"
    encoder.parent.mkdir(parents=True)
    encoder.symlink_to(outside)
    result = subprocess.run([sys.executable, "-m", "src.tennis_scene.scripts.preflight",
        f"paths.data_root={tmp_path / 'data'}", f"paths.checkpoint_root={checkpoint}",
        f"paths.external_asset_root={tmp_path / 'third_party'}", f"paths.output_root={tmp_path / 'outputs'}"],
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "PYTHONPATH": "",
                              "TENNIS_LAB_DINO_EXTENSION": ""},
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    checks = {check["name"]: check for check in report["checks"]}
    assert report["inference_executed"] is False
    assert report["failed"] == sum(check["status"] != "ok" for check in report["checks"])
    assert {name: checks[name]["status"] for name in ("court", "ball", "tracking_encoder", "association_encoder",
                                                   "vitpose", "video_cam0", "dino_extension")} == {
        "court": "invalid_path", "ball": "mismatch", "tracking_encoder": "invalid_path",
        "association_encoder": "invalid_path", "vitpose": "missing", "video_cam0": "missing", "dino_extension": "error"}
    assert "unavailable" in checks["pipeline_configuration"]["detail"]
    assert outside.read_bytes() == b"outside checkpoint must remain untouched"


def test_invalid_root_reports_why_inventory_is_unavailable(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "src.tennis_scene.scripts.preflight",
        "paths.checkpoint_root=/", f"paths.output_root={tmp_path / 'outputs'}"],
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["inference_executed"] is False
    assert report["failed"] == 1
    assert report["checks"][0]["name"] == "pipeline_configuration"
    assert "Asset inventory unavailable" in report["checks"][0]["detail"]


def test_broken_association_yaml_and_unsafe_checkpoint_still_produce_json(tmp_path: Path) -> None:
    project = tmp_path / "project"
    association = project / "src/tasks/player_association/configs/association_i964_r14_lovo_a.yaml"
    association.parent.mkdir(parents=True)
    association.write_text("appearance: [invalid YAML")
    checkpoint = tmp_path / "ckpt"
    court = checkpoint / "court_detection/multiscale_depth3/b863df1f01f0.ckpt"
    court.parent.mkdir(parents=True)
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"outside must not be read")
    court.symlink_to(outside)
    result = subprocess.run([sys.executable, "-m", "src.tennis_scene.scripts.preflight",
        f"paths.project_root={project}", f"paths.data_root={tmp_path / 'data'}",
        f"paths.checkpoint_root={checkpoint}", f"paths.external_asset_root={tmp_path / 'third_party'}",
        f"paths.output_root={tmp_path / 'outputs'}"],
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    checks = {check["name"]: check for check in report["checks"]}
    assert report["inference_executed"] is False
    assert checks["court"]["status"] == "invalid_path"
    assert checks["ball"]["status"] == "missing"
    assert checks["association_configuration"]["status"] == "error"
    assert "Association encoder inventory unavailable" in checks["association_configuration"]["detail"]
    assert "association_encoder" not in checks
