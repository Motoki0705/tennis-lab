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
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert report["inference_executed"] is False
    failures = {check["name"] for check in report["checks"] if check["status"] != "ok"}
    assert {"court", "ball", "detector", "vitpose", "hmr2", "gvhmr", "aflink", "tracking_encoder",
            "association_encoder", "body_model", "dinov3", "smpl_faces", "dino_models", "dinov3_hub",
            "smpl_hmr2_mean_params", "smpl_smplx_to_smpl", "smpl_smpl_coco17_regressor",
            "smpl_smplx_verts437", "smpl_smpl_neutral_joint_regressor", "video_cam0", "video_cam1", "video_cam2"} <= failures
