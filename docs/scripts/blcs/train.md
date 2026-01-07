# BLCS train

2Dボール位置シーケンスから3Dボール軌道を推定するモデルを学習するスクリプト。

## 概要

このスクリプトは、カメラ視点での2Dボール位置シーケンスとコートキーポイントから、3D空間でのボール軌道を推定するTransformerベースのモデルを学習します。PyTorch Lightning を使用し、TensorBoard によるロギング、チェックポイント保存、Early Stopping などをサポートします。

## コマンド例

```bash
# デフォルト設定で学習
uv run python -m src.blcs.scripts.train

# エポック数とGPU設定を指定
uv run python -m src.blcs.scripts.train training.max_epochs=5 run.gpus=0

# 高速デバッグモード
uv run python -m src.blcs.scripts.train run.fast_dev_run=true

# 学習を再開
uv run python -m src.blcs.scripts.train run.resume=outputs/blcs/single/logs/version_0/checkpoints/last.ckpt

# バッチサイズと学習率を変更
uv run python -m src.blcs.scripts.train data.batch_size=64 training.learning_rate=5e-5
```

## コンフィグ

エントリポイント: `src/blcs/configs/train.yaml`

### defaults 構成

```yaml
defaults:
  - model: default
  - data: default
  - training: default
  - metrics: default
  - run: train
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `outputs/blcs/single` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `gpus` | `1` | 使用するGPU数 (0=CPU) |
| `fast_dev_run` | `false` | デバッグ用高速実行 |
| `resume` | `null` | 再開するチェックポイントパス |

### model (モデル設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `name` | `blcs` | モデル名 |
| `hidden_dim` | `256` | 隠れ層の次元数 |
| `num_layers` | `6` | Transformer層の数 |
| `num_heads` | `8` | Attention ヘッド数 |
| `dropout` | `0.1` | ドロップアウト率 |
| `max_seq_len` | `120` | 最大シーケンス長 (30fps × 4秒) |
| `use_cross_attention` | `true` | コートキーポイントへのクロスアテンション |
| `ball_input_dim` | `2` | ボール入力次元 (u, v) |
| `court_kp_dim` | `40` | コートキーポイント次元 (20点 × 2) |
| `position_dim` | `3` | 出力位置次元 (x, y, z) |

### data (データ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `scene_dir` | `data/blcs` | シーンデータのディレクトリ |
| `batch_size` | `32` | バッチサイズ |
| `num_workers` | `4` | データローダーのワーカー数 |
| `val_split` | `0.1` | 検証データの割合 |
| `test_split` | `0.1` | テストデータの割合 |
| `min_seq_len` | `15` | 最小シーケンス長 (0.5秒) |
| `max_seq_len` | `120` | 最大シーケンス長 (4秒) |
| `fps` | `30` | フレームレート |
| `camera_mode` | `random` | カメラ選択モード |
| `augmentation.uv_noise_std` | `0.005` | UV座標ノイズの標準偏差 |
| `augmentation.visibility_drop_prob` | `0.1` | 可視性ドロップ確率 |
| `augmentation.temporal_dropout_prob` | `0.05` | 時間方向ドロップアウト確率 |
| `augmentation.flip_horizontal` | `true` | 水平反転 |
| `augmentation.scale_range` | `[0.9, 1.1]` | スケール範囲 |

### training (学習設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_epochs` | `200` | 最大エポック数 |
| `learning_rate` | `1.0e-4` | 学習率 |
| `weight_decay` | `1.0e-5` | 重み減衰 |
| `warmup_steps` | `2000` | ウォームアップステップ数 |
| `gradient_clip_val` | `1.0` | 勾配クリッピング値 |
| `position_loss_weight` | `1.0` | 位置ロスの重み |
| `velocity_loss_weight` | `0.0` | 速度ロスの重み |
| `smoothness_loss_weight` | `0.0` | 滑らかさロスの重み |
| `scheduler` | `cosine` | 学習率スケジューラ |
| `min_lr` | `1.0e-6` | 最小学習率 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              train.py                                        │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  BLCSDataModule │──────▶│BLCSLightningMod │──────▶│    pl.Trainer       │  │
│  │                 │      │                 │      │                     │  │
│  │ - Scene読込    │      │ - BLCS Model    │      │ - GPU/CPU 管理      │  │
│  │ - Sequence抽出 │      │ - Position Loss │      │ - Checkpoint        │  │
│  │ - Augmentation │      │ - Metrics       │      │ - EarlyStopping     │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

入力:
  - ball_uv: [B, T, 2]        # 正規化された2Dボール位置
  - court_kp: [B, 20, 2]      # 正規化された2Dコートキーポイント
  - visibility: [B, T]        # 可視性マスク

出力:
  - ball_pos_3d: [B, T, 3]    # 正規化された3D位置 (x, y, z)
```

## モデルアーキテクチャ

```
Input: ball_uv [B, T, 2] + court_kp [B, 20, 2]
                    │
                    ▼
            ┌───────────────┐
            │  Embedding    │
            │  ball → d_model
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  Positional   │
            │  Encoding     │
            └───────────────┘
                    │
                    ▼
     ┌──────────────┴──────────────┐
     │         Transformer          │
     │  ┌─────────────────────────┐│
     │  │    Self-Attention       ││
     │  │    (temporal)           ││
     │  └─────────────────────────┘│
     │              │               │
     │  ┌─────────────────────────┐│
     │  │   Cross-Attention       ││
     │  │   (ball → court_kp)     ││
     │  └─────────────────────────┘│
     │              │               │
     │  ┌─────────────────────────┐│
     │  │      FFN Layer          ││
     │  └─────────────────────────┘│
     │         × N layers          │
     └─────────────┬───────────────┘
                   │
                   ▼
            ┌───────────────┐
            │   MLP Head    │
            │   → [B, T, 3] │
            └───────────────┘
```

## 出力構造

```
outputs/blcs/single/
├── config.yaml              # 使用した設定
├── checkpoints/
│   ├── blcs-epoch=XX.ckpt   # ベストモデル
│   └── last.ckpt            # 最終モデル
└── logs/
    └── version_X/
        └── events.out.tfevents.*
```

## 評価メトリクス

- `position_error_m`: 3D位置の平均誤差 (メートル)
- `val/loss`: 検証損失

## 関連モジュール

- `src.blcs.data.datamodule`: データモジュール
- `src.blcs.training.lightning_module`: Lightning モジュール
- `src.blcs.models`: モデル定義
