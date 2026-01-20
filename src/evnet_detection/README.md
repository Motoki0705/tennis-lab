# evnet_detection

Train event detection models using the BLCS-generated dataset.

Models:
- **UV model**: predicts shot/bounce timings from `ball_uv` + `court_kp` (for scene segmentation).
- **3D model**: predicts shot/bounce timings from `ball_pos_world` only (for within-scene event timing).

Entry points:
- `uv run python -m src.evnet_detection.scripts.train_uv`
- `uv run python -m src.evnet_detection.scripts.train_3d`

