"""SLCS (Scene Localization in Court System).

Temporal multitask model that fuses per-camera 2D observations (player pose,
ball, court keypoints) with sparsely sampled DINOv3 patch tokens and predicts
player 3D position/yaw, ball 3D position and per-output uncertainty in court
coordinates, trained on the structured real-clip dataset defined by issue #634
(`dataset.json` / `clip.json` / `annotations/tennis_scene/scene.npz`).
"""
