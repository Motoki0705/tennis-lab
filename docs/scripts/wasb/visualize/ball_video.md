# WASB visualize/ball_video

学習済みモデルを使ってボール位置を検出し、オーバーレイ動画を生成するスクリプト。

## 概要

このスクリプトは、入力動画に対してWASB/HRCNetモデルで推論を行い、検出されたボール位置をオーバーレイした出力動画を生成します。

## コマンド例

```bash
# 基本的な使用法
uv run python -m src.tasks.wasb.scripts.visualize.ball_video video_path=data/tennis/raw/videos/match.mp4

# モデルとチェックポイントを指定
uv run python -m src.tasks.wasb.scripts.visualize.ball_video \
  video_path=data/samples/clip.mp4 \
  checkpoint=outputs/wasb/ball_detection/hrcnet/logs/version_0/checkpoints/last.ckpt \
  model=hrcnet \
  device=cuda

# 出力パスを指定
uv run python -m src.tasks.wasb.scripts.visualize.ball_video \
  video_path=input.mp4 \
  output_path=output_with_ball.mp4

# フレーム数を制限（テスト用）
uv run python -m src.tasks.wasb.scripts.visualize.ball_video \
  video_path=... \
  max_frames=300
```

## コンフィグ

エントリポイント: `src/tasks/wasb/configs/plot_ball_video.yaml`

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `video_path` | `data/samples/clip.mp4` | 入力動画パス |
| `output_path` | `outputs/wasb/ball_detection/visualize/ball_video/clip.mp4` | 出力動画パス |
| `checkpoint` | (WASB pretrained) | モデルチェックポイント |
| `model` | `wasb` | モデル名 (wasb/hrcnet) |
| `device` | `cuda` | デバイス |
| `batch_size` | `64` | バッチサイズ |
| `max_frames` | `null` | 最大フレーム数 |
| `score_threshold` | `0.5` | 検出スコア閾値 |

### render (レンダリング設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `radius` | `6` | ボールマーカーの半径 |
| `thickness` | `-1` | マーカーの線幅 (-1=塗りつぶし) |
| `color_detected_bgr` | `[0, 255, 0]` | 検出ボールの色 (緑) |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ball_video.py                                      │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  Input Video    │──────▶│ WASB/HRCNet     │──────▶│   VideoBall         │  │
│  │                 │      │ Predictor       │      │   Localization       │  │
│  │ - cv2.VideoCapture     │ - バッチ推論    │      │   Pipeline           │  │
│  │ - フレーム抽出  │      │ - ヒートマップ  │      │   → xy_px, visibility│  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
│                                                               │              │
│                                                               ▼              │
│                                               ┌─────────────────────────────┐│
│                                               │   Overlay Renderer          ││
│                                               │                             ││
│                                               │ - フレーム読み直し          ││
│                                               │ - cv2.circle でオーバーレイ ││
│                                               │ - cv2.VideoWriter で出力    ││
│                                               └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. 入力動画をバッチでフレーム抽出
2. WASB/HRCNet でボール位置を検出
3. フレームを再読み込みし、ボール位置をオーバーレイ
4. 出力動画を書き出し
```

## visibility_code の値

| 値 | 意味 | 表示色 |
|----|------|--------|
| `0` | 不可視（検出なし） | 表示なし |
| `1` | 検出（モデル出力） | 緑 |

## 出力例

```
outputs/wasb/ball_detection/visualize/ball_video/
└── clip.mp4    # ボール位置オーバーレイ動画
```

## 関連モジュール

- `src.tasks.wasb.inference.ball_detection.WASBPredictor`: WASB 推論
- `src.tasks.wasb.inference.ball_detection.HRCNetWASBPredictor`: HRCNet 推論
- `src.tasks.wasb.pipeline.video_ball_localization_pipeline.VideoBallLocalizationPipeline`: パイプライン
