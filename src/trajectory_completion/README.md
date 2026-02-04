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
outputs = predictor.predict(
    ball_uv_in,
    ball_obs_mask,  # visibility (observed=1, missing=0)
    court_kp,
    court_vis=court_vis,
)
completed = outputs["ball_uv_completed"]
```

## Dataset output keys (breaking change)

`BLCSUVTrajectoryCompletionDataset` now returns visibility signals with clear semantics:

- `ball_vis`: ground-truth visibility (1=visible, 0=invisible/invalid).
- `ball_obs_mask`: observed mask after corruption (1=observed, 0=missing).

If you previously used `ball_vis` as the observed mask, update your training
and inference pipelines to consume `ball_obs_mask` instead.

The dataset also accepts `split` (train/val/test) or `split_file` for scene
resolution. Provide only one of them per dataset instance.

## Visualization

Dataset-side visualization (how much GT is masked / how much observed points jitter):

- `uv run python -m src.trajectory_completion.scripts.visualize`
- `uv run python -m src.trajectory_completion.scripts.visualize visualization.scene_path=data/blcs/scenes/rally_000000.npz`
- Corruption tuning (reproducible with `run.seed`):
    - `uv run python -m src.trajectory_completion.scripts.visualize run.seed=0 data.corruption.noise_std=0.02 data.corruption.point_dropout_prob=0.2`

Inference visualization (distinguishes predictions at observed vs masked frames):

- `uv run python -m src.trajectory_completion.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/trajectory_completion/.../last.ckpt`
- Save the figure:
    - `uv run python -m src.trajectory_completion.scripts.visualize visualization.save=outputs/tmp/vis.png`

## Data loading optimization

- `data.cache_max_scenes`: LRU cache size for scene NPZ files.
- `data.scene_sampler_mode`: `none | scene | mixed | chunked`.
- `data.chunk_max_scenes`: Number of scenes per chunk for chunked sampling.
