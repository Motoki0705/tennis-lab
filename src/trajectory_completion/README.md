# trajectory_completion

Train and run inference for UV trajectory completion using court keypoints.

Entry points:
- `uv run python -m src.trajectory_completion.scripts.train`

Inference:
- `src/trajectory_completion/inference/uv_predictor.py`

Example:
```python
from src.trajectory_completion.inference import UVTrajectoryCompletionPredictor

predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint("path/to/checkpoint.ckpt")
outputs = predictor.predict(ball_uv_in, ball_obs_mask, court_kp, court_vis=court_vis)
completed = outputs["ball_uv_completed"]
```
