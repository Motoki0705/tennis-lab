# ball_detection

`ball_detection` is the task package for supervised and semi-supervised ball
detection training based on the spatio-temporal U-Net experiment under
`experiments/ball_detection`.

## Scope

- Included now:
  - Hydra training entrypoint for supervised training
  - task-local config groups
  - supervised dataset and manual dataloader construction
  - STUNet model migration
  - supervised manual PyTorch training loop
  - phase-based semi-supervised training from the same `train.py` entrypoint
  - raw-video-based pseudo-label generation utilities under `training/`
  - public module boundaries for `models`, `data`, `training`, and `inference`
- Deferred to later phases:
  - predictor implementation

## Design Decisions

- `training.semi_supervised.num_semi_phases=0` keeps the run purely supervised.
- `training.semi_supervised.num_semi_phases>0` enables phase-based pseudo-label generation.
- Split management will use manifest files rather than in-dataset random splitting.
- Default `sample_stride` is `4` to match the current experiment implementation.
- Pseudo-labeling reads `data/tennis/raw/videos/video_*.mp4` directly.
- Decoded pseudo frames are stored once under `data/tennis/pseudo_label/cache/` and
  reused across later phases.
- Each phase writes only phase-specific manifests and pseudo labels under
  `data/tennis/pseudo_label/phase_XX/`.
- The batch contract will be sequence-first at the dataset boundary:
  - `images`: `(B, T, 3, H, W)`
  - `heatmaps`: `(B, T, H/2, W/2)`
  - `coords`: `(B, T, 2)` in original-image pixel coordinates
  - `visibility`: `(B, T)`
- Target heatmaps use continuous Gaussian values rather than binary disks.
- Metric distance thresholds are evaluated in original-image pixels.
- GPU training uses `deterministic: warn` because STUNet contains a
  `max_pool3d` backward path without a deterministic CUDA kernel in PyTorch.

## Planned Layout

```text
src/tasks/ball_detection/
├── configs/
├── data/
├── inference/
├── models/
├── scripts/
└── training/
```

## Commands

Supervised training entrypoint:

```bash
uv run python -m src.tasks.ball_detection.scripts.train
```

Semi-supervised training from the same entrypoint:

```bash
uv run python -m src.tasks.ball_detection.scripts.train \
  training.semi_supervised.num_semi_phases=1 \
  training.semi_supervised.phase0_epochs=15 \
  training.semi_supervised.phase_epochs=10
```

Download raw videos listed in `src/tasks/ball_detection/configs/downlaod.yaml`:

```bash
uv run python -m src.tasks.ball_detection.scripts.download_videos
```

Downloaded videos are stored under `data/tennis/raw/videos` as `video_<n>.mp4`,
and `data/tennis/raw/videos/summary.json` records the mapping from source URL to
the renamed local file.
