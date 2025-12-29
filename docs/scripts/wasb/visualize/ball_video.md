# WASB visualize/ball_video

学習済みモデルを使ってボール位置を検出し、オーバーレイ動画を生成するスクリプト。

## 概要

このスクリプトは、入力動画に対してWASB/HRCNetモデルで推論を行い、検出されたボール位置をオーバーレイした出力動画を生成します。オプションで軌道補完も適用できます。

## コマンド例

```bash
# 基本的な使用法
uv run python -m src.wasb.scripts.visualize.ball_video video_path=data/tennis/raw/videos/match.mp4

# モデルとチェックポイントを指定
uv run python -m src.wasb.scripts.visualize.ball_video \
  video_path=data/samples/clip.mp4 \
  checkpoint=outputs/wasb/checkpoints/last.ckpt \
  model=hrcnet \
  device=cuda

# 軌道補完を有効化
uv run python -m src.wasb.scripts.visualize.ball_video \
  video_path=... \
  completion.enabled=true \
  completion.method=bilstm \
  completion.checkpoint_path=outputs/trajectory/checkpoints/last.ckpt

# 出力パスを指定
uv run python -m src.wasb.scripts.visualize.ball_video \
  video_path=input.mp4 \
  output_path=output_with_ball.mp4

# フレーム数を制限（テスト用）
uv run python -m src.wasb.scripts.visualize.ball_video \
  video_path=... \
  max_frames=300
```

## コンフィグ

エントリポイント: `src/wasb/configs/plot_ball_video.yaml`

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `video_path` | `data/samples/clip.mp4` | 入力動画パス |
| `output_path` | `outputs/plot_ball_video/clip.mp4` | 出力動画パス |
| `checkpoint` | (WASB pretrained) | モデルチェックポイント |
| `model` | `wasb` | モデル名 (wasb/hrcnet) |
| `device` | `cuda` | デバイス |
| `batch_size` | `64` | バッチサイズ |
| `max_frames` | `null` | 最大フレーム数 |
| `score_threshold` | `0.5` | 検出スコア閾値 |

### completion (軌道補完設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enabled` | `true` | 補完を有効化 |
| `method` | `bilstm` | 補完方法 (physics/bilstm/transformer/refiner/hybrid) |
| `checkpoint_path` | (trajectory model) | 補完モデルのチェックポイント |
| `device` | `${device}` | デバイス |
| `max_gap` | `15` | 補完する最大ギャップ |
| `physics_gap_threshold` | `5` | 物理ベース補完の閾値 |

### render (レンダリング設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_completion` | `true` | 補完結果を表示するか |
| `radius` | `6` | ボールマーカーの半径 |
| `thickness` | `-1` | マーカーの線幅 (-1=塗りつぶし) |
| `color_detected_bgr` | `[0, 255, 0]` | 検出ボールの色 (緑) |
| `color_completed_bgr` | `[0, 255, 255]` | 補完ボールの色 (黄) |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ball_video.py                                      │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  Input Video    │──────▶│ WASB/HRCNet     │──────▶│   Completer         │  │
│  │                 │      │ Predictor       │      │   (optional)        │  │
│  │ - cv2.VideoCapture     │ - バッチ推論    │      │ - BiLSTM等          │  │
│  │ - フレーム抽出  │      │ - ヒートマップ  │      │ - ギャップ補完      │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
│                                                               │              │
│                                                               ▼              │
│                                               ┌─────────────────────────────┐│
│                                               │   VideoBallLocalization     ││
│                                               │   Pipeline                  ││
│                                               │                             ││
│                                               │ → xy_px, visibility_code    ││
│                                               └─────────────────────────────┘│
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
3. (completion.enabled=true) 軌道補完を適用
4. フレームを再読み込みし、ボール位置をオーバーレイ
5. 出力動画を書き出し
```

## visibility_code の値

| 値 | 意味 | 表示色 |
|----|------|--------|
| `0` | 不可視（検出なし） | 表示なし |
| `1` | 検出（モデル出力） | 緑 |
| `2` | 補完（軌道補完） | 黄 |

## 出力例

```
outputs/plot_ball_video/
└── clip.mp4    # ボール位置オーバーレイ動画
```

## 補完方法

| 方法 | 説明 |
|------|------|
| `physics` | 物理ベースの補間（放物線） |
| `bilstm` | BiLSTM モデルによる補完 |
| `transformer` | Transformer モデルによる補完 |
| `refiner` | 反復精緻化モデルによる補完 |
| `hybrid` | 短いギャップは物理、長いギャップはモデル |

## 関連モジュール

- `src.wasb.inference.WASBPredictor`: WASB 推論
- `src.wasb.inference.HRCNetWASBPredictor`: HRCNet 推論
- `src.wasb.inference.build_completer`: 補完器の構築
- `src.wasb.pipeline.VideoBallLocalizationPipeline`: パイプライン
