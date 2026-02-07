# event_detection

Train event detection models using the BLCS-generated dataset.

Models:
- **UV model**: predicts shot/bounce timings from `ball_uv` + `court_kp` (for scene segmentation).
- **3D model**: predicts shot/bounce timings from `ball_pos_world` only (for within-scene event timing).

Entry points:
- `uv run python -m src.event_detection.scripts.train` (UV model, default)
- `uv run python -m src.event_detection.scripts.train --config-name train_3d` (3D model)

Inference:
- `src/event_detection/inference/uv_predictor.py` for UV-based event detection
- `src/event_detection/inference/traj3d_predictor.py` for 3D-trajectory event detection

Example:
```python
from src.event_detection.inference import UVEventPredictor

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

GTラベル（soft targets）と推論結果（prob/peaks）を単一シーンで確認できます。

- UV:
    - `uv run python -m src.event_detection.scripts.visualize_uv visualization.scene_path=data/blcs/scenes/rally_000000.npz`
    - 推論: `uv run python -m src.event_detection.scripts.visualize_uv visualization.mode=predict visualization.checkpoint=...`
- 3D:
    - `uv run python -m src.event_detection.scripts.visualize_3d visualization.scene_path=data/blcs/scenes/rally_000000.npz`
    - 推論: `uv run python -m src.event_detection.scripts.visualize_3d visualization.mode=predict visualization.checkpoint=...`

Animation（`visualization.view=animation`）では、event発生フレームでボール色が変化します（GT/予測どちらも反映）。

## Data loading optimization

- `data.cache_max_scenes`: LRU cache size for scene NPZ files.
- `data.scene_sampler_mode`: `none | scene | mixed | chunked`.
- `data.chunk_max_scenes`: Number of scenes per chunk for chunked sampling.
