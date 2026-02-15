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
    ball_uv,
    court_kp,
    ball_vis=ball_vis,    # observed mask (1=observed, 0=missing)
    ball_mask=ball_mask,  # optional padding/valid mask (1=valid)
    court_vis=court_vis,
)
completed = outputs["ball_uv_completed"]
```

## Dataset output keys (breaking change)

`BLCSUVTrajectoryCompletionDataset` now returns visibility signals with clear semantics:

- `ball_uv`: corrupted UV input for the model.
- `ball_vis`: observed mask after augmentation (1=observed, 0=missing) for model input.
- `ball_uv_gt`: ground-truth UV.
- `ball_gt_vis`: ground-truth visibility (1=visible, 0=invisible/invalid) for supervision.
- `seq_len`: sequence length used by collate to build `ball_mask`.

If you previously used `ball_uv_in` / `ball_obs_mask` / old predictor signature,
update call sites to the BLCS-like order:
`predict(ball_uv, court_kp, ball_vis=None, ball_mask=None, court_vis=None)`.

The dataset also accepts `split` (train/val/test) or `split_file` for scene
resolution. Provide only one of them per dataset instance.

## Visualization

Animation-only visualization:

- `uv run python -m src.trajectory_completion.scripts.visualize`
- `uv run python -m src.trajectory_completion.scripts.visualize visualization.scene_path=data/blcs/scenes/rally_000000.npz`
- Inference animation (GT line + predicted point, color-coded by observed/completed frames):
    - `uv run python -m src.trajectory_completion.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/trajectory_completion/.../last.ckpt`
- Save animation:
    - `uv run python -m src.trajectory_completion.scripts.visualize visualization.save=outputs/tmp/vis.gif`

## Training design: observed drift mitigation

`src/trajectory_completion/training/lightning_module.py` supports two mechanisms to reduce
drift propagation from `observed` to `masked`.

- Masked-loss schedule:
    - `training.loss.masked_schedule.enabled`
    - `training.loss.masked_schedule.start_epoch`
    - `training.loss.masked_schedule.end_epoch`
    - `training.loss.masked_schedule.weight_min`
    - `training.loss.masked_schedule.weight_max`
- Auxiliary observed loss on intermediate layers:
    - `training.loss.auxiliary_observed.enabled`
    - `training.loss.auxiliary_observed.weight`
    - `training.loss.auxiliary_observed.depth_weighting` (`linear` or `exp`)
    - `training.loss.auxiliary_observed.exp_gamma`
    - `training.loss.auxiliary_observed.predictor_hidden_dim`

Recommended recipe:
- Early training: keep masked weight small (`weight_min`) and optimize observed + smoothness.
- Later training: ramp masked weight to `weight_max`.
- Keep auxiliary observed loss enabled to retain token-local coordinate information.

New logs:
- `train/val masked_weight_t`: effective masked loss weight after scheduling.
- `train/val loss_aux`: summed auxiliary observed loss (before global aux scaling).
- `train/val boundary_jump_pred`, `boundary_jump_gt`, `boundary_jump_error`: boundary jump diagnostics at observed/masked transitions.

## Data loading optimization

- `data.cache_max_scenes`: LRU cache size for scene NPZ files.
- `data.scene_sampler_mode`: `none | scene | mixed | chunked`.
- `data.chunk_max_scenes`: Number of scenes per chunk for chunked sampling.
