# WASB visualize/ball_video_ensemble

複数モデルのアンサンブルでボール位置を検出し、オーバーレイ動画を生成するスクリプト。

## 概要

このスクリプトは、複数の学習済みWASBモデル（HRNet, HRCNet, DINOv3など）のヒートマップ出力をアンサンブルして、より頑健なボール検出を行います。

## コマンド例

```bash
# 基本的な使用法
uv run python -m src.wasb.scripts.visualize.ball_video_ensemble video_path=data/samples/clip.mp4

# ヒートマップ閾値を変更
uv run python -m src.wasb.scripts.visualize.ball_video_ensemble \
  video_path=... \
  ensemble.heatmap_threshold=0.5

# カスタムチェックポイントを指定
uv run python -m src.wasb.scripts.visualize.ball_video_ensemble \
  video_path=... \
  'ensemble.checkpoints=[a.ckpt,b.ckpt,c.ckpt]'

# 出力パスを指定
uv run python -m src.wasb.scripts.visualize.ball_video_ensemble \
  video_path=input.mp4 \
  output_path=output_ensemble.mp4
```

## コンフィグ

エントリポイント: `src/wasb/configs/plot_ball_video_ensemble.yaml`

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `video_path` | `data/samples/clip.mp4` | 入力動画パス |
| `output_path` | `outputs/plot_ball_video_ensemble/clip.mp4` | 出力動画パス |
| `device` | `cpu` | デバイス |
| `batch_size` | `64` | バッチサイズ |
| `max_frames` | `null` | 最大フレーム数 |

### ensemble (アンサンブル設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `checkpoints` | (5つのモデル) | チェックポイントのリスト |
| `heatmap_threshold` | `0.5` | ヒートマップ閾値 |
| `apply_sigmoid` | `true` | シグモイドを適用するか |
| `output_heatmap_hw` | `null` | 出力ヒートマップサイズ（リサイズ用） |

デフォルトのチェックポイント構成例:
```yaml
checkpoints:
  - outputs/hrcnet_frames3/logs/version_0/checkpoints/last.ckpt
  - outputs/hrcnet_frames5/logs/version_0/checkpoints/last.ckpt
  - outputs/hrnet_frames3/logs/version_0/checkpoints/last.ckpt
  - outputs/hrnet_frames5/logs/version_0/checkpoints/last.ckpt
  - outputs/dinov3_heatmap_frames1/logs/version_0/checkpoints/last.ckpt
```

### completion (軌道補完設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enabled` | `false` | 補完を有効化 |
| `method` | `bilstm` | 補完方法 |
| `checkpoint_path` | (trajectory model) | 補完モデルパス |
| `device` | `${device}` | デバイス |
| `max_gap` | `15` | 最大ギャップ |
| `physics_gap_threshold` | `5` | 物理補完閾値 |

### render (レンダリング設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_completion` | `true` | 補完結果を表示 |
| `radius` | `6` | マーカー半径 |
| `thickness` | `-1` | 線幅 |
| `color_detected_bgr` | `[0, 255, 0]` | 検出色 |
| `color_completed_bgr` | `[0, 255, 255]` | 補完色 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ball_video_ensemble.py                                  │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  Input Video    │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   HeatmapEnsemblePredictor                          │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────┐│    │
│  │  │ HRCNet-3  │ │ HRCNet-5  │ │ HRNet-3   │ │ HRNet-5   │ │DINOv3  ││    │
│  │  │ frames    │ │ frames    │ │ frames    │ │ frames    │ │        ││    │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬───┘│    │
│  │        │             │             │             │            │    │    │
│  │        └──────────────┴──────────────┴──────────────┴───────────┘    │    │
│  │                                    │                                 │    │
│  │                                    ▼                                 │    │
│  │                        ┌─────────────────────┐                       │    │
│  │                        │   Heatmap Average   │                       │    │
│  │                        │   (Ensemble)        │                       │    │
│  │                        └─────────────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │   Peak Detection    │                             │
│                         │   → (x, y) position │                             │
│                         └─────────────────────┘                             │
│                                    │                                        │
│                                    ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │   Overlay Renderer  │                             │
│                         └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. 各モデルでフレームを処理し、ヒートマップを生成
2. ヒートマップを同一サイズにリサイズ（必要に応じて）
3. ヒートマップの平均（または他のアグリゲーション）を計算
4. 閾値以上のピークを検出してボール位置を決定
5. 検出結果をオーバーレイして動画出力
```

## アンサンブルの利点

| 単一モデル | アンサンブル |
|-----------|-------------|
| ✗ 特定条件で弱い | ✓ 複数モデルで補完 |
| ✗ 偽陽性が発生しやすい | ✓ 複数モデルの合意で抑制 |
| ✗ 入力フレーム数固定 | ✓ 異なるフレーム数を組み合わせ |
| ✓ 高速 | ✗ 遅い（複数モデル実行） |

## 推奨するアンサンブル構成

1. **時間解像度の多様性**: 異なる `frames_in` を持つモデル (3, 5, 8 など)
2. **アーキテクチャの多様性**: HRNet, HRCNet, DINOv3 など
3. **奇数個のモデル**: 多数決の際に有利

## 出力構造

```
outputs/plot_ball_video_ensemble/
└── clip.mp4    # アンサンブル検出結果のオーバーレイ動画
```

## 関連モジュール

- `src.wasb.inference.HeatmapEnsemblePredictor`: アンサンブル推論
- `src.wasb.pipeline.VideoBallLocalizationPipeline`: パイプライン
