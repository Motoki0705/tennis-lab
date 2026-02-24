# ball_detection

`src/tasks/ball_detection` implements a staged workflow for tennis ball detection:

1. supervised pretraining on labeled data
2. pseudo-label generation on unlabeled data via ensemble + refinement
3. self-training on mixed labeled/pseudo data

## Key structure

- `data/io`: dataset layout, annotation ingestion/merge, and writing policies
- `pseudo/components`: clip sampling, trajectory refinement, event tagging, quality checks
- `pseudo/orchestrator.py`: end-to-end pseudo-label generation workflow
- `training`: Lightning training modules and runner
- `inference/`: 推論設定・predictor・heatmapアンサンブル・入力アダプター
- `visualization/`: 可視化モジュール群（`api` / `adapters` / `io` / `rendering` / `analysis` / `orchestrator`）
- `scripts`: Hydra entrypoints

## Third-party backbones

- `third_party/WASB-SBDT/src/models/hrnet.py` is used as the HRNet backbone source.
- `third_party/TrackNetV3` is added as a git submodule for TrackNetV3 fine-tuning.

## Pretrained checkpoint download

```bash
uv run python -m src.tasks.ball_detection.scripts.download_pretrained --models all
```

This downloads:

- `checkpoints/wasb/wasb_tennis_best.pth.tar`
- `checkpoints/tracknetv3/TrackNet_best.pt`
- `checkpoints/tracknetv3/InpaintNet_best.pt`

## Fine-tuning presets

- HRNet backbone + temporal ConvGRU heatmap training:

```bash
uv run python -m src.tasks.ball_detection.scripts.train_pretrain --config-name train_pretrain_hrnet
```

- TrackNetV3 heatmap fine-tuning:

```bash
uv run python -m src.tasks.ball_detection.scripts.train_pretrain --config-name train_pretrain_tracknetv3
```

Both presets keep heatmap generation inside `src/tasks/ball_detection` training logic.

## 可視化（single / ensemble 切替）

Hydra 設定で `inference.strategy=single|ensemble` を切り替えられます。

```bash
# デフォルト: ensemble（ball_detection ckpt群）
uv run python -m src.tasks.ball_detection.scripts.visualize

# 単体推論
uv run python -m src.tasks.ball_detection.scripts.visualize inference.strategy=single

# 動画・出力先の上書き
uv run python -m src.tasks.ball_detection.scripts.visualize \
    visualization.video_path=data/samples/test.mp4 \
    visualization.output_video_path=outputs/ball_detection/visualize/test_ball_overlay.mp4
```
