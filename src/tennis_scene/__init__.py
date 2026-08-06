"""Tennis Scene 3D Reconstruction Module.

This module provides an integrated pipeline for reconstructing 3D tennis scenes
from monocular video, combining:
- GVHMR: 3D human mesh (SMPL) estimation
- Ball detection: 2D ball detection from `src.tasks.ball_detection`
- Court KP Detection: 2D court keypoint detection
- PLCS: 3D player localization in court system
- BLCS: 3D ball localization in court system

Fixed camera assumption:
- Court keypoints estimated from a single frame
- No camera rotation estimation (static_cam=True for GVHMR)
- GVHMR provides local SMPL only
- PLCS position and yaw are applied to SMPL mesh

The pipeline uses an orchestration pattern with modular components in
``src.tennis_scene.pipeline``. Import schemas, archive functions, and the
composition root from their owning modules; this package does not mirror those
APIs under additional names.
"""
