# WASB train/trajectory

ボール軌道の補完（trajectory completion）を行うモデルを学習するスクリプト。

## 概要

このスクリプトは、検出されたボール位置シーケンスの欠損部分を補完するモデルを学習します。BiLSTM、Transformer、Refiner などのアーキテクチャをサポートし、オクルージョンやミス検出による欠損を予測して埋めます。

## コマンド例

```bash
# デフォルト設定で学習
uv run python -m src.wasb.scripts.train.trajectory

# エポック数とGPU設定を指定
uv run python -m src.wasb.scripts.train.trajectory training.max_epochs=1 run.gpus=0

# ドライランモード
uv run python -m src.wasb.scripts.train.trajectory run.dry_run=true

# モデルを変更
uv run python -m src.wasb.scripts.train.trajectory model=trajectory_transformer

# シーケンス長を変更
uv run python -m src.wasb.scripts.train.trajectory data.sequence_length=128
```

## コンフィグ

エントリポイント: `src/wasb/configs/train_trajectory.yaml`

### defaults 構成

```yaml
defaults:
  - data: trajectory
  - training: trajectory
  - loss: trajectory
  - logging: default
  - metrics: trajectory
  - run: trajectory
  - model: trajectory_bilstm
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `outputs/wasb/trajectory` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `gpus` | `1` | 使用するGPU数 |
| `fast_dev_run` | `false` | デバッグ用高速実行 |
| `dry_run` | `false` | データ確認のみ |

### model (モデル設定)

#### trajectory_bilstm (デフォルト)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `name` | `trajectory_bilstm` | モデル名 |
| `hidden_dim` | `64` | 隠れ層の次元数 |
| `num_layers` | `2` | LSTM 層数 |
| `dropout` | `0.1` | ドロップアウト率 |
| `score_threshold` | `0.5` | スコア閾値 |

#### trajectory_transformer

Transformer ベースの軌道補完モデル。

#### trajectory_refiner

反復的な軌道精緻化を行うモデル。

### data (データ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `root_dir` | `data/tennis` | データセットルート |
| `train_matches` | `["game1",...,"game8"]` | 学習データのマッチ |
| `val_matches` | `["game9"]` | 検証データのマッチ |
| `test_matches` | `["game10"]` | テストデータのマッチ |
| `sequence_length` | `64` | シーケンス長 |
| `step` | `8` | ウィンドウのステップ |
| `min_visible_per_window` | `4` | ウィンドウあたりの最小可視フレーム |
| `block_mask_min_len` | `3` | ブロックマスクの最小長 |
| `block_mask_max_len` | `5` | ブロックマスクの最大長 |
| `sparse_mask_prob` | `0.1` | スパースマスクの確率 |
| `noise_prob` | `0.1` | ノイズ付加確率 |
| `noise_std_px` | `3.0` | ノイズの標準偏差 (ピクセル) |
| `batch_size` | `32` | バッチサイズ |
| `num_workers` | `4` | データローダーのワーカー数 |

### training (学習設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_epochs` | `50` | 最大エポック数 |
| `learning_rate` | `1.0e-3` | 学習率 |
| `weight_decay` | `1.0e-4` | 重み減衰 |
| `warmup_steps` | `1000` | ウォームアップステップ数 |
| `min_lr` | `1.0e-6` | 最小学習率 |
| `lambda_block` | `1.0` | ブロック補完ロスの重み |
| `lambda_sparse` | `1.0` | スパース補完ロスの重み |
| `lambda_noise` | `1.0` | ノイズ除去ロスの重み |
| `precision` | `32` | 精度 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            trajectory.py                                     │
│                                                                              │
│  ┌───────────────────┐    ┌───────────────────┐    ┌─────────────────────┐  │
│  │TrajectoryDataModul│────▶│TrajectoryLightning│────▶│    pl.Trainer       │  │
│  │                   │    │                   │    │                     │  │
│  │ - xy_input (部分) │    │ - BiLSTM/Trans.  │    │ - GPU/CPU 管理      │  │
│  │ - target_xy (全体)│    │ - Completion     │    │ - Checkpoint        │  │
│  │ - マスク生成      │    │ - Loss計算        │    │ - Logging           │  │
│  └───────────────────┘    └───────────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

入力:
  - xy_input_norm: [B, T, 2]    # 部分的に欠損した正規化座標
  - loss_mask_block: [B, T]     # ブロック欠損マスク
  - loss_mask_sparse: [B, T]    # スパース欠損マスク
  - loss_mask_noise: [B, T]     # ノイズ付加マスク

出力:
  - pred_xy_norm: [B, T, 2]     # 補完された正規化座標
```

## マスク戦略

学習時に3種類のマスク戦略を適用：

### 1. Block Mask
連続した区間を欠損させる。オクルージョンをシミュレート。
```
Original:  ● ● ● ● ● ● ● ● ● ●
Masked:    ● ● □ □ □ □ ● ● ● ●
```

### 2. Sparse Mask
ランダムなフレームを欠損させる。散発的なミス検出をシミュレート。
```
Original:  ● ● ● ● ● ● ● ● ● ●
Masked:    ● ● □ ● ● □ ● □ ● ●
```

### 3. Noise Mask
ノイズを付加。検出誤差をシミュレート。
```
Original:  ● ● ● ● ● ●
Noisy:     ● ●̃ ● ●̃ ● ●  (̃ = ノイズ付加)
```

## BiLSTM モデルアーキテクチャ

```
Input: xy_input [B, T, 2]
            │
            ▼
    ┌───────────────┐
    │   Embedding   │
    │   2 → hidden  │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │   BiLSTM      │
    │               │
    │  Forward  →   │
    │  ← Backward   │
    │               │
    │  × N layers   │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │   Output Head │
    │   → [B, T, 2] │
    └───────────────┘
```

## 出力構造

```
outputs/wasb/trajectory/trajectory_bilstm/
├── config.yaml
├── logs/
│   └── version_X/
│       ├── checkpoints/
│       │   ├── trajectory-epoch=XX.ckpt
│       │   └── last.ckpt
│       ├── events.out.tfevents.*
│       └── vis/               # 可視化結果
│           ├── sample_0.png
│           └── ...
```

## 評価メトリクス

- `val/loss`: 検証損失
- 距離誤差 (ピクセル)

## 関連モジュール

- `src.wasb.data.trajectory_datamodule`: データモジュール
- `src.wasb.training.TrajectoryLightningModule`: Lightning モジュール
- `src.wasb.models.trajectory.bilstm`: BiLSTM モデル
- `src.wasb.models.trajectory.transformer`: Transformer モデル
