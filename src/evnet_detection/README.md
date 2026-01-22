# evnet_detection

Train event detection models using the BLCS-generated dataset.

Models:
- **UV model**: predicts shot/bounce timings from `ball_uv` + `court_kp` (for scene segmentation).
- **3D model**: predicts shot/bounce timings from `ball_pos_world` only (for within-scene event timing).

Entry points:
- `uv run python -m src.evnet_detection.scripts.train_uv`
- `uv run python -m src.evnet_detection.scripts.train_3d`

Inference:
- `src/evnet_detection/inference/uv_predictor.py` for UV-based event detection
- `src/evnet_detection/inference/traj3d_predictor.py` for 3D-trajectory event detection

Example:
```python
from src.evnet_detection.inference import UVEventPredictor

predictor = UVEventPredictor.load_from_checkpoint("path/to/checkpoint.ckpt", device="cpu")
outputs = predictor.predict(
    ball_uv,
    court_kp,
    ball_vis=ball_vis,  # visibility (1=visible)
    ball_mask=ball_mask,  # padding mask (1=valid)
    court_vis=court_vis,
)
event_peaks = outputs["event_peaks"]  # list[B][E][N]
```
