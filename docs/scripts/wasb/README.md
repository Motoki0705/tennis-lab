# WASB Scripts

Where's the Ball (WASB) タスクのスクリプト群。

テニス映像からボール位置を検出し、軌道補完やイベント検出を行います。

## スクリプト一覧

### 学習スクリプト (`train/`)

| スクリプト | 説明 | ドキュメント |
|-----------|------|-------------|
| `ball_detection` | ボール検出モデルの学習 | [train/ball_detection.md](train/ball_detection.md) |
| `trajectory` | 軌道補完モデルの学習 | [train/trajectory.md](train/trajectory.md) |
| `event_detection` | イベント検出モデルの学習 | [train/event_detection.md](train/event_detection.md) |

### データセット生成スクリプト (`generate_dataset/`)

| スクリプト | 説明 | ドキュメント |
|-----------|------|-------------|
| `batch` (メイン) | 動画からデータセット生成 | [generate_dataset/batch.md](generate_dataset/batch.md) |
| `clip_sampling` | クリップのプレビュー・選別 | [generate_dataset/clip_sampling.md](generate_dataset/clip_sampling.md) |
| `download_videos` | YouTube動画のダウンロード | [generate_dataset/download_videos.md](generate_dataset/download_videos.md) |

### 可視化スクリプト (`visualize/`)

| スクリプト | 説明 | ドキュメント |
|-----------|------|-------------|
| `ball_video` | ボール検出結果のオーバーレイ動画 | [visualize/ball_video.md](visualize/ball_video.md) |
| `ball_video_ensemble` | アンサンブル検出のオーバーレイ動画 | [visualize/ball_video_ensemble.md](visualize/ball_video_ensemble.md) |
| `trajectory` | 軌道補完結果の可視化 | [visualize/trajectory.md](visualize/trajectory.md) |
| `save_one_sample_visuals` | データセットサンプルの確認 | [visualize/save_one_sample_visuals.md](visualize/save_one_sample_visuals.md) |

### ツールスクリプト (`tools/`)

| スクリプト | 説明 | ドキュメント |
|-----------|------|-------------|
| `extract_dinov3_backbone` | DINOv3バックボーン重みの抽出 | [tools/extract_dinov3_backbone.md](tools/extract_dinov3_backbone.md) |
| `encode_dinov3_patch_tokens` | パッチトークンの事前計算 | [tools/encode_dinov3_patch_tokens.md](tools/encode_dinov3_patch_tokens.md) |

## 典型的なワークフロー

### A. 新規データセット作成

```bash
# 1. 動画をダウンロード
uv run python -m src.tasks.wasb.scripts.generate_dataset.download_videos

# 2. ボール位置をアノテーション
uv run python -m src.tasks.wasb.scripts.generate_dataset mode=batch

# 3. プレビュー生成と手動選別
uv run python -m src.tasks.wasb.scripts.generate_dataset.clip_sampling mode=generate_samples generate_samples=[all]
# （不要なプレビューを手動削除）
uv run python -m src.tasks.wasb.scripts.generate_dataset.clip_sampling mode=apply_clip_selection apply_clip_selection=[all]
```

### B. モデル学習

```bash
# 1. ボール検出モデル
uv run python -m src.tasks.wasb.scripts.train.ball_detection

# 2. 軌道補完モデル
uv run python -m src.tasks.wasb.scripts.train.trajectory

# 3. イベント検出モデル
uv run python -m src.tasks.wasb.scripts.train.event_detection
```

### C. 推論・可視化

```bash
# ボール検出結果を動画に出力
uv run python -m src.tasks.wasb.scripts.visualize.ball_video \
  video_path=data/samples/clip.mp4 \
  checkpoint=outputs/wasb/ball_detection/hrcnet/logs/version_0/checkpoints/last.ckpt
```

## ディレクトリ構成

```
src/tasks/wasb/
├── scripts/
│   ├── train/
│   │   ├── ball_detection.py
│   │   ├── trajectory.py
│   │   └── event_detection.py
│   ├── generate_dataset/
│   │   ├── __main__.py
│   │   ├── batch.py
│   │   ├── clip_sampling.py
│   │   └── download_videos.py
│   ├── visualize/
│   │   ├── ball_video.py
│   │   ├── ball_video_ensemble.py
│   │   ├── trajectory.py
│   │   └── save_one_sample_visuals.py
│   └── tools/
│       ├── extract_dinov3_backbone.py
│       └── encode_dinov3_patch_tokens.py
├── configs/
│   ├── train_ball_detection.yaml
│   ├── train_trajectory.yaml
│   ├── train_event_detection.yaml
│   ├── generate_dataset.yaml
│   ├── plot_ball_video.yaml
│   └── ...
├── data/
├── models/
├── training/
├── inference/
├── pipeline/
└── tennis_format/
```
