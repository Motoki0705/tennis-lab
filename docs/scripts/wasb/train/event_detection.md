# WASB train/event_detection

ボール軌道上のイベント（ショット、バウンス）を検出するモデルを学習するスクリプト。

## 概要

このスクリプトは、ボールの2D座標シーケンスから、各フレームでのイベント（ショット、バウンス、なし）を分類するTransformerベースのモデルを学習します。

## コマンド例

```bash
# デフォルト設定で学習
uv run python -m src.wasb.scripts.train.event_detection

# ドライランモード
uv run python -m src.wasb.scripts.train.event_detection run.dry_run=true

# エポック数とGPU設定を指定
uv run python -m src.wasb.scripts.train.event_detection training.max_epochs=1 run.gpus=0

# シーケンス長を変更
uv run python -m src.wasb.scripts.train.event_detection data.sequence_length=256
```

## コンフィグ

エントリポイント: `src/wasb/configs/train_event_detection.yaml`

### defaults 構成

```yaml
defaults:
  - data: event_detection
  - training: event_detection
  - loss: event_detection
  - logging: default
  - metrics: event_detection
  - run: event_detection
  - model: event_detection_transformer
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `outputs/event_detection` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `gpus` | `0` | 使用するGPU数 |
| `fast_dev_run` | `false` | デバッグ用高速実行 |
| `dry_run` | `false` | データ確認のみ |

### model (モデル設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `name` | `event_detection_transformer` | モデル名 |

### data (データ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `root_dir` | `data/tennis` | データセットルート |
| `train_matches` | `["game1",...,"game8"]` | 学習データのマッチ |
| `val_matches` | `["game9"]` | 検証データのマッチ |
| `test_matches` | `["game10"]` | テストデータのマッチ |
| `sequence_length` | `128` | シーケンス長 |
| `step` | `8` | ウィンドウのステップ |
| `min_visible_per_window` | `8` | ウィンドウあたりの最小可視フレーム |
| `xy_scale` | `[1920.0, 1080.0]` | XY座標のスケール |
| `ignore_invisible_targets` | `true` | 不可視フレームのターゲットを無視 |
| `ignore_index` | `-100` | 無視するインデックス |
| `batch_size` | `64` | バッチサイズ |
| `num_workers` | `4` | データローダーのワーカー数 |

### training (学習設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_epochs` | `50` | 最大エポック数 |
| `learning_rate` | `3.0e-4` | 学習率 |
| `weight_decay` | `1.0e-4` | 重み減衰 |
| `warmup_steps` | `1000` | ウォームアップステップ数 |
| `min_lr` | `1.0e-6` | 最小学習率 |
| `precision` | `32` | 精度 |
| `ignore_index` | `-100` | CrossEntropyLoss で無視するインデックス |
| `class_weights` | `null` | 手動クラス重み（null = 自動計算） |
| `compute_class_weights.enabled` | `true` | クラス重みを自動計算 |
| `compute_class_weights.max_windows` | `5000` | 重み計算に使うウィンドウ数 |
| `background_weight_scale` | `0.02` | 背景クラスの重みスケール |
| `event_boost` | `2.0` | イベントクラスのブースト係数 |
| `label_smoothing` | `0.0` | ラベルスムージング |

## イベントクラス

| クラス | 値 | 説明 |
|--------|-----|------|
| `none` | `0` | イベントなし（背景） |
| `shot` | `1` | ショット（ラケット打球） |
| `bounce` | `2` | バウンス（コート接地） |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         event_detection.py                                   │
│                                                                              │
│  ┌───────────────────┐    ┌───────────────────┐    ┌─────────────────────┐  │
│  │TrajectoryEventData│────▶│EventDetectionLight│────▶│    pl.Trainer       │  │
│  │                   │    │                   │    │                     │  │
│  │ - xy_norm         │    │ - Transformer     │    │ - GPU/CPU 管理      │  │
│  │ - target_status   │    │ - 3クラス分類     │    │ - Checkpoint        │  │
│  │ - visibility      │    │ - CrossEntropy    │    │ - Logging           │  │
│  └───────────────────┘    └───────────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

入力:
  - xy_norm: [B, T, 2]         # 正規化された2D座標
  - visibility: [B, T]         # 可視性マスク

出力:
  - logits: [B, T, 3]          # 各フレームの3クラス分類ロジット
```

## クラスバランス対策

イベント検出はクラス不均衡問題（背景が大半）のため、以下の対策を実装：

1. **逆頻度重み付け**: クラス頻度の逆数で CrossEntropyLoss を重み付け
2. **背景重みスケール**: 背景クラスの重みをさらに小さく（0.02倍）
3. **イベントブースト**: イベントクラスのロスを強調（2.0倍）

```
# 例: 推定クラス分布
none (0):   ~95%  → weight: 0.05 × 0.02 = 0.001
shot (1):   ~2%   → weight: 0.5  × 2.0  = 1.0
bounce (2): ~3%   → weight: 0.33 × 2.0  = 0.66
```

## 出力構造

```
outputs/event_detection/event_detection_transformer/
├── config.yaml
├── logs/
│   └── version_X/
│       ├── checkpoints/
│       │   ├── event-epoch=XX.ckpt
│       │   └── last.ckpt
│       └── events.out.tfevents.*
```

## 評価メトリクス

- `val/loss`: 検証損失
- `accuracy`: 全体精度
- `precision_shot`: ショット検出精度
- `recall_shot`: ショット検出再現率
- `precision_bounce`: バウンス検出精度
- `recall_bounce`: バウンス検出再現率

## 関連モジュール

- `src.wasb.data.event_detection_datamodule`: データモジュール
- `src.wasb.training.EventDetectionLightningModule`: Lightning モジュール
- `src.wasb.models.event_detection_transformer`: Transformer モデル
