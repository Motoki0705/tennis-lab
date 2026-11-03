"""Pin the current demo export choices; presentation fills do not revive ball gaps."""
import argparse
from pathlib import Path

from src.tennis_scene.visualization.blender.export.bundle import BlenderBundleConfig, export_blender_bundle

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=Path, required=True)
parser.add_argument('--checkpoints', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
print(export_blender_bundle(
    scene_path=args.scene.resolve(),
    smpl_faces_path=args.checkpoints / 'body_models/smplh/neutral/model.npz',
    smpl_joint_regressor_path=args.checkpoints / 'body_models/gvhmr/smpl_neutral_J_regressor.pt',
    output_dir=args.output.resolve(),
    config=BlenderBundleConfig(camera_id='cam1', player_max_gap_frames=12, ball_max_gap_frames=0,
        max_ball_reprojection_median_px=8., sensor_width_mm=36., plate_jpeg_quality=95),
))
