# WASB train/ball_detection

画面上のボール位置を検出するためのヒートマップベースモデルを学習するスクリプト。

## 概要

このスクリプトは、テニス映像からボールの2D位置を検出するモデルを学習します。DINOv3 ViTバックボーンにFPNを組み合わせたヒートマップ予測モデルや、HRNet/HRCNet ベースのモデルをサポートします。

## コマンド例

```bash
# デフォルト設定で学習
uv run python -m src.wasb.scripts.train.ball_detection

# エポック数とバッチサイズを変更
uv run python -m src.wasb.scripts.train.ball_detection training.max_epochs=50 data.batch_size=32

# ドライランモード（データ確認用）
uv run python -m src.wasb.scripts.train.ball_detection run.dry_run=true

# CPU で実行
uv run python -m src.wasb.scripts.train.ball_detection run.gpus=0

# モデルを変更
uv run python -m src.wasb.scripts.train.ball_detection model=hrcnet

# 高速デバッグモード
uv run python -m src.wasb.scripts.train.ball_detection run.fast_dev_run=true
```

## コンフィグ

エントリポイント: `src/wasb/configs/train_ball_detection.yaml`

### defaults 構成

```yaml
defaults:
  - data: ball_detection
  - training: ball_detection
  - loss: ball_detection
  - logging: default
  - metrics: ball_detection
  - run: ball_detection
  - model: dinov3_heatmap
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `outputs/wasb/ball_detection` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `gpus` | `1` | 使用するGPU数 (0=CPU) |
| `fast_dev_run` | `false` | デバッグ用高速実行 |
| `dry_run` | `false` | データ確認のみ |

### model (モデル設定)

#### dinov3_heatmap (デフォルト)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `name` | `dinov3_heatmap` | モデル名 |
| `fpn_channels` | `[256, 128, 64, 32]` | FPN チャンネル数 |
| `backbone_checkpoint` | (DINOv3 pretrained) | バックボーンの事前学習済み重み |

#### hrcnet / hrnet

HRNet/HRCNet ベースのモデルも選択可能です。

### data (データ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `root_dir` | `data/tennis` | データセットルート |
| `train_matches` | `["game1",...,"game7"]` | 学習データのマッチ |
| `val_matches` | `["game8", "game9"]` | 検証データのマッチ |
| `test_matches` | `["game10"]` | テストデータのマッチ |
| `frames_in` | `8` | 入力フレーム数 |
| `frames_out` | `1` | 出力フレーム数 |
| `step` | `1` | フレームステップ |
| `visibility_mode` | `all_visible` | 可視性モード |
| `batch_size` | `4` | バッチサイズ |
| `num_workers` | `4` | データローダーのワーカー数 |
| `resize_hw` | `[288, 512]` | リサイズ後のサイズ |
| `heatmap_hw` | `[288, 512]` | ヒートマップサイズ |
| `heatmap_sigma` | `2.0` | ガウシアンヒートマップのσ |

### data.augment (データ拡張設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enabled` | `true` | 拡張を有効化 |
| `color_jitter.prob` | `0.8` | Color Jitter の確率 |
| `color_jitter.brightness` | `0.5` | 明度変動 |
| `random_grayscale.prob` | `0.3` | グレースケール変換確率 |
| `gaussian_blur.prob` | `0.8` | ガウシアンブラー確率 |
| `random_erasing.prob` | `0.2` | Random Erasing 確率 |

### data.sampling (サンプリング設定)

| パラメータ | 説明 |
|-----------|------|
| `curriculum.enabled` | カリキュラム学習を有効化 |
| `curriculum.switch_step` | カリキュラム切り替えステップ |
| `curriculum.balance_visibility` | バランスさせる可視性クラス |
| `curriculum.target_ratio` | 目標比率 |

### training (学習設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_epochs` | `20` | 最大エポック数 |
| `freeze_backbone_epochs` | `10` | バックボーン凍結エポック数 |
| `learning_rate` | `1e-4` | 学習率 |
| `backbone_learning_rate` | `1e-5` | バックボーン学習率 |
| `weight_decay` | `1e-4` | 重み減衰 |
| `warmup_steps` | `1000` | ウォームアップステップ数 |
| `bce_weight` | `1.0` | BCE ロスの重み |
| `mse_weight` | `0.0` | MSE ロスの重み |
| `temporal_weight` | `0.1` | 時間整合性ロスの重み |
| `precision` | `bf16-mixed` | 混合精度訓練 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ball_detection.py                                    │
│                                                                              │
│  ┌───────────────────┐    ┌───────────────────┐    ┌─────────────────────┐  │
│  │BallDetectionDataMo│────▶│ WASBLightningMod  │────▶│    pl.Trainer       │  │
│  │                   │    │                   │    │                     │  │
│  │ - フレームシーケン │    │ - DINOv3+FPN     │    │ - GPU/CPU 管理      │  │
│  │   ス読み込み      │    │ - Heatmap出力     │    │ - Checkpoint        │  │
│  │ - Heatmap生成    │    │ - BCE Loss        │    │ - Logging           │  │
│  │ - Augmentation   │    │ - Metrics         │    │                     │  │
│  └───────────────────┘    └───────────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

入力:
  - frames: [B, T, C, H, W]  # T フレームの画像シーケンス

出力:
  - heatmaps: [B, T_out, H', W']  # ボール位置のヒートマップ
```

## モデルアーキテクチャ（DINOv3 + FPN）

```
Input: frames [B, T, 3, H, W]
            │
            ▼
    ┌───────────────┐
    │   DINOv3 ViT  │  (pretrained)
    │   Backbone    │
    │               │
    │  - patch_embed│
    │  - transformer│
    │    layers     │
    └───────────────┘
            │
            ▼ multi-scale features
    ┌───────────────┐
    │      FPN      │
    │               │
    │  - 256 ch     │
    │  - 128 ch     │
    │  - 64 ch      │
    │  - 32 ch      │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  Heatmap Head │
    │    → [H, W]   │
    └───────────────┘
```

## 出力構造

```
outputs/wasb/ball_detection/dinov3_heatmap/
├── config.yaml              # 使用した設定
├── logs/
│   └── version_X/
│       ├── checkpoints/
│       │   ├── wasb-epoch=XX.ckpt
│       │   └── last.ckpt
│       ├── events.out.tfevents.*
│       └── dry_run/         # (dry_run=true の場合)
│           ├── frame_XX.png
│           ├── heatmap_XX.png
│           └── overlay_XX.png
```

## 評価メトリクス

- `val/loss`: 検証損失
- `precision`: ボール検出精度
- `recall`: ボール検出再現率
- `distance_error`: 予測と正解の距離誤差

## 関連モジュール

- `src.wasb.data.ball_detection_datamodule`: データモジュール
- `src.wasb.training.WASBLightningModule`: Lightning モジュール
- `src.wasb.models.dinov3_fpn_heatmap`: DINOv3+FPN モデル
- `src.wasb.models.hrcnet`: HRCNet モデル
