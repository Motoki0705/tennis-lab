# PLCS train_sequence

シーケンス単位でプレーヤーの3D位置・回転を推定するモデルを学習するスクリプト。

## 概要

このスクリプトは、複数フレームのシーケンスを入力として時系列情報を活用し、より正確な3D位置・回転推定を行うモデルを学習します。`train.py` と同じ `run_training` 関数を使用しますが、シーケンス用のコンフィグを使用します。

## コマンド例

```bash
# デフォルト設定で学習
uv run python -m src.plcs.scripts.train_sequence

# GPU設定とエポック数を指定
uv run python -m src.plcs.scripts.train_sequence run.gpus=1 training.max_epochs=50

# シーケンス長を変更
uv run python -m src.plcs.scripts.train_sequence data.seq_len=64

# 高速デバッグモード
uv run python -m src.plcs.scripts.train_sequence run.fast_dev_run=true
```

## コンフィグ

エントリポイント: `src/plcs/configs/train_sequence.yaml`

### defaults 構成

```yaml
defaults:
  - model: sequence
  - data: sequence
  - training: default
  - loss: sequence
  - metrics: default
  - run: train_sequence
```

### loss (ロス設定)

シーケンスモデルでは時間一貫性ロスが有効になります。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `position_weight` | `1.0` | 位置ロスの重み |
| `rotation_weight` | `1.0` | 回転ロスの重み |
| `temporal_weight` | `0.1` | 時間一貫性ロスの重み |
| `temporal.order` | `2` | 時間微分の次数（2=加速度平滑化） |
| `temporal.robust` | `true` | SmoothL1Lossを使用 |

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `outputs/plcs_sequence` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `gpus` | `1` | 使用するGPU数 |
| `fast_dev_run` | `false` | デバッグ用高速実行 |
| `resume` | `null` | 再開するチェックポイントパス |

### model (モデル設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `name` | `plcs_sequence` | モデル名 |
| `hidden_dim` | `256` | 隠れ層の次元数 |
| `num_layers` | `8` | Transformer層の数 |
| `num_heads` | `8` | Attention ヘッド数 |
| `dropout` | `0.1` | ドロップアウト率 |
| `max_seq_len` | `120` | 最大シーケンス長 |
| `architecture` | `sequence` | アーキテクチャタイプ |

### data (データ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `scene_dir` | `data/plcs/scenes` | シーンデータのディレクトリ |
| `batch_size` | `64` | バッチサイズ |
| `num_workers` | `4` | データローダーのワーカー数 |
| `val_split` | `0.1` | 検証データの割合 |
| `test_split` | `0.1` | テストデータの割合 |
| `camera_mode` | `all` | カメラ選択モード |
| `mode` | `sequence` | データモード |
| `seq_len` | `32` | シーケンス長 |
| `seq_stride` | `8` | シーケンスのストライド |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          train_sequence.py                                   │
│                                                                              │
│  ┌───────────────────┐    ┌───────────────────┐    ┌─────────────────────┐  │
│  │PLCSSequenceDataMod│────▶│PLCSSequenceLightni│────▶│    pl.Trainer       │  │
│  │                   │    │                   │    │                     │  │
│  │ - Sequence抽出   │    │ - Sequence Model  │    │ - GPU/CPU 管理      │  │
│  │ - Temporal Aug.  │    │ - Temporal Loss   │    │ - Checkpoint        │  │
│  │ - Sliding Window │    │ - Metrics         │    │ - EarlyStopping     │  │
│  └───────────────────┘    └───────────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

入力 (シーケンス):
  - human_keypoints: [B, T, 17, 2]  # T フレームの人物キーポイント
  - court_keypoints: [B, T, 20, 2]  # T フレームのコートキーポイント

出力:
  - position: [B, T, 3]   # 各フレームの3D位置
  - rotation: [B, T, 2]   # 各フレームの回転
```

## フレームモデル vs シーケンスモデル

| 項目 | frame モデル | sequence モデル |
|------|-------------|----------------|
| 入力 | 1フレーム | T フレーム |
| 時系列情報 | なし | あり |
| Transformer層 | 4 | 8 |
| メモリ使用量 | 少 | 多 |
| 推論速度 | 速い | 遅い |
| 精度 | 基準 | 高い（時系列一貫性） |

## 出力構造

```
outputs/plcs_sequence/
├── config.yaml                  # 使用した設定
├── checkpoints/
│   ├── plcs-seq-epoch=XX.ckpt   # ベストモデル
│   └── last.ckpt                # 最終モデル
└── logs/
    └── version_X/
        └── events.out.tfevents.*
```

## 関連モジュール

- `src.plcs.data.datamodule.PLCSSequenceDataModule`: シーケンスデータモジュール
- `src.plcs.training.sequence_lightning_module`: シーケンス用 Lightning モジュール
