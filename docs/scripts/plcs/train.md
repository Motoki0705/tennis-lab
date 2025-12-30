# PLCS train

フレーム単位でプレーヤーの3D位置・回転を推定するモデルを学習するスクリプト。

## 概要

このスクリプトは、2Dキーポイント（人物・コート）から3Dプレーヤー位置と回転を推定するTransformerベースのモデルを学習します。PyTorch Lightning を使用し、TensorBoard によるロギング、チェックポイント保存、Early Stopping などをサポートします。

## コマンド例

```bash
# デフォルト設定で学習
uv run python -m src.plcs.scripts.train

# GPU 設定とエポック数を指定
uv run python -m src.plcs.scripts.train run.gpus=0 training.max_epochs=1

# 高速デバッグモード
uv run python -m src.plcs.scripts.train run.fast_dev_run=true

# 学習を再開
uv run python -m src.plcs.scripts.train run.resume=outputs/plcs/checkpoints/last.ckpt

# バッチサイズを変更
uv run python -m src.plcs.scripts.train data.batch_size=128
```

## コンフィグ

エントリポイント: `src/plcs/configs/train.yaml`

### defaults 構成

```yaml
defaults:
  - model: frame
  - data: frame
  - training: default
  - loss: frame
  - metrics: default
  - run: train
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `outputs/plcs` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `gpus` | `1` | 使用するGPU数 (0=CPU) |
| `fast_dev_run` | `false` | デバッグ用高速実行 |
| `resume` | `null` | 再開するチェックポイントパス |

### model (モデル設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `name` | `plcs` | モデル名 |
| `hidden_dim` | `256` | 隠れ層の次元数 |
| `num_layers` | `4` | Transformer層の数 |
| `num_heads` | `8` | Attention ヘッド数 |
| `dropout` | `0.1` | ドロップアウト率 |
| `use_court_context` | `true` | コートキーポイントを使用するか |
| `human_kp_dim` | `34` | 人物キーポイント次元 (17点 × 2) |
| `court_kp_dim` | `40` | コートキーポイント次元 (20点 × 2) |
| `position_dim` | `3` | 出力位置次元 (x, y, z) |
| `rotation_dim` | `2` | 出力回転次元 (sin, cos of yaw) |

### data (データ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `scene_dir` | `data/plcs/scenes` | シーンデータのディレクトリ |
| `batch_size` | `64` | バッチサイズ |
| `num_workers` | `4` | データローダーのワーカー数 |
| `val_split` | `0.1` | 検証データの割合 |
| `test_split` | `0.1` | テストデータの割合 |
| `camera_mode` | `random` | カメラ選択モード |
| `mode` | `frame` | データモード (frame/sequence) |
| `augmentation.keypoint_noise_std` | `0.01` | キーポイントノイズの標準偏差 |
| `augmentation.visibility_drop_prob` | `0.05` | 可視性ドロップ確率 |

### training (学習設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_epochs` | `100` | 最大エポック数 |
| `learning_rate` | `1.0e-4` | 学習率 |
| `weight_decay` | `1.0e-5` | 重み減衰 |
| `warmup_steps` | `1000` | ウォームアップステップ数 |
| `gradient_clip_val` | `1.0` | 勾配クリッピング値 |
| `scheduler` | `cosine` | 学習率スケジューラ |
| `min_lr` | `1.0e-6` | 最小学習率 |

### loss (ロス設定)

ロス設定は `src/plcs/configs/loss/` ディレクトリに分離されています。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `position_weight` | `1.0` | 位置ロスの重み |
| `rotation_weight` | `1.0` | 回転ロスの重み |
| `temporal_weight` | `0.0` | 時間一貫性ロスの重み（フレームモードでは0） |
| `temporal.order` | `2` | 時間微分の次数（1=速度、2=加速度） |
| `temporal.robust` | `true` | SmoothL1Lossを使用するか |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              train.py                                        │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  PLCSDataModule │──────▶│PLCSLightningMod │──────▶│   pl.Trainer        │  │
│  │                 │      │                 │      │                     │  │
│  │ - Scene NPZ読込│      │ - PLCS Model    │      │ - GPU/CPU 管理      │  │
│  │ - Augmentation │      │ - Loss計算      │      │ - Checkpoint        │  │
│  │ - Split管理    │      │ - Metrics       │      │ - EarlyStopping     │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

入力 (1フレーム):
  - human_keypoints: [B, 17, 2]  # 正規化された2D人物キーポイント
  - court_keypoints: [B, 20, 2]  # 正規化された2Dコートキーポイント

出力:
  - position: [B, 3]   # 正規化された3D位置 (x, y, z)
  - rotation: [B, 2]   # 回転 (sin(yaw), cos(yaw))
```

## 出力構造

```
outputs/plcs/
├── config.yaml              # 使用した設定
├── checkpoints/
│   ├── plcs-epoch=XX.ckpt   # ベストモデル
│   └── last.ckpt            # 最終モデル
└── logs/
    └── version_X/
        └── events.out.tfevents.*  # TensorBoard ログ
```

## 関連モジュール

- `src.plcs.data.datamodule`: データモジュール
- `src.plcs.training.lightning_module`: Lightning モジュール
- `src.plcs.models`: モデル定義
