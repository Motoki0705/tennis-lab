# event_detection

Train event detection models using the BLCS-generated dataset.

Models:
- **UV model**: predicts shot/bounce timings from `ball_uv` + `court_kp` (for scene segmentation).
- **3D model**: predicts shot/bounce timings from `ball_pos_world` only (for within-scene event timing).

Entry points:
- `uv run python -m src.tasks.event_detection.scripts.train_uv`
- `uv run python -m src.tasks.event_detection.scripts.train_3d`

Inference:
- `src/tasks/event_detection/inference/uv_predictor.py` for UV-based event detection
- `src/tasks/event_detection/inference/traj3d_predictor.py` for 3D-trajectory event detection

Example:
```python
from src.tasks.event_detection.inference import UVEventPredictor

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

## Visualization

単一シーンのアニメーション可視化のみを提供します。
軌道はGT系列を使用し、イベント表現は推論結果（predicted peaks）のみを使用します。

- UV:
    - `uv run python -m src.tasks.event_detection.scripts.visualize visualization.scene_path=data/blcs/scenes/rally_000000.npz`
    - 推論: `uv run python -m src.tasks.event_detection.scripts.visualize visualization.mode=predict visualization.checkpoint=...`
- 3D:
    - `uv run python -m src.tasks.event_detection.scripts.visualize visualization=traj3d data=blcs_rally_3d visualization.scene_path=data/blcs/scenes/rally_000000.npz`
    - 推論: `uv run python -m src.tasks.event_detection.scripts.visualize visualization=traj3d data=blcs_rally_3d visualization.mode=predict visualization.checkpoint=...`

Animationでは、推論イベント周辺フレームでもボール色が連続的に変化します。

## Data loading optimization

- `data.cache_max_scenes`: LRU cache size for scene NPZ files.
- `data.scene_sampler_mode`: `none | scene | mixed | chunked`.
- `data.chunk_max_scenes`: Number of scenes per chunk for chunked sampling.
