# ball_detection

`src/ball_detection` implements a staged workflow for tennis ball detection:

1. supervised pretraining on labeled data
2. pseudo-label generation on unlabeled data via ensemble + refinement
3. self-training on mixed labeled/pseudo data

## Key structure

- `data/io`: dataset layout, annotation ingestion/merge, and writing policies
- `pseudo/components`: clip sampling, trajectory refinement, event tagging, quality checks
- `pseudo/orchestrator.py`: end-to-end pseudo-label generation workflow
- `training`: Lightning training modules and runner
- `scripts`: Hydra entrypoints

## Third-party backbones

- `third_party/WASB-SBDT/src/models/hrnet.py` is used as the HRNet backbone source.
- `third_party/TrackNetV3` is added as a git submodule for TrackNetV3 fine-tuning.

## Pretrained checkpoint download

```bash
uv run python -m src.ball_detection.scripts.download_pretrained --models all
```

This downloads:

- `checkpoints/wasb/wasb_tennis_best.pth.tar`
- `checkpoints/tracknetv3/TrackNet_best.pt`
- `checkpoints/tracknetv3/InpaintNet_best.pt`

## Fine-tuning presets

- HRNet backbone + temporal ConvGRU heatmap training:

```bash
uv run python -m src.ball_detection.scripts.train_pretrain --config-name train_pretrain_hrnet
```

- TrackNetV3 heatmap fine-tuning:

```bash
uv run python -m src.ball_detection.scripts.train_pretrain --config-name train_pretrain_tracknetv3
```

Both presets keep heatmap generation inside `src/ball_detection` training logic.
