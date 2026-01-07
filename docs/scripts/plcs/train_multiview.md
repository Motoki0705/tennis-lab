# PLCS train_multiview.py

マルチビュー（複数カメラ）からの観測を使用してプレイヤーの3D位置・回転を推定するモデルを学習するスクリプトです。

## 概要

`train_multiview.py` は、複数カメラからの2D人物キーポイントとコートキーポイントを入力として、
コート座標系でのプレイヤーの3D位置と回転（向き）を推定するモデルを学習します。

**シーケンシャル入力に対応**: 複数カメラ×時系列 `(N_cam, T, ...)` の入力をサポートし、
時間一貫性ロスによるスムーズな軌道推定が可能です。

単一カメラの `train.py` とは異なり、複数視点の情報を統合することで、
遮蔽やカメラ配置の制約を克服し、より堅牢な推定を実現します。

## 実行方法

### 基本実行

```bash
uv run python -m src.plcs.scripts.train_multiview
```

### 出力先・データセット指定

```bash
uv run python -m src.plcs.scripts.train_multiview \
    run.output_dir=outputs/plcs/multiview \
    data.scene_dir=data/plcs/scenes
```

### カメラ数の設定

```bash
# 4カメラ同時使用、最低2カメラ必須
uv run python -m src.plcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2
```

### シーケンス長の設定

```bash
# シーケンス長を32フレームに設定
uv run python -m src.plcs.scripts.train_multiview \
    data.seq_len=32
```

### ランダムレンジサンプリング

```bash
# カメラ数とシーケンス長を範囲からランダムにサンプリング
uv run python -m src.plcs.scripts.train_multiview \
    'data.num_views_range=[1, 8]' \
    'data.seq_len_range=[4, 32]'
```

### Dry Run（データローディング確認のみ）

```bash
uv run python -m src.plcs.scripts.train_multiview run.dry_run=true
```

### GPU学習

```bash
uv run python -m src.plcs.scripts.train_multiview run.gpus=1
```

### 高速開発モード（1バッチのみ）

```bash
uv run python -m src.plcs.scripts.train_multiview run.fast_dev_run=true
```

## 設定ファイル構成

### メイン設定

`src/plcs/configs/train_multiview.yaml`:

```yaml
defaults:
  - data: multiview
  - model: multiview
  - training: default
  - run: train_multiview

run:
  seed: 42
  gpus: 0
  output_dir: outputs/plcs/multiview
  dry_run: false
  fast_dev_run: false

hydra:
  run:
    dir: ${run.output_dir}
```

### データ設定

`src/plcs/configs/data/multiview.yaml`:

```yaml
scene_dir: data/plcs
batch_size: 32
num_workers: 4
num_views: 2          # 同時に使用するカメラ数
min_cameras: 2        # シーンに必要な最小カメラ数
seq_len: 16           # シーケンス長

# ランダムレンジサンプリング（オプション）
# num_views_range: [1, 8]   # 各サンプルでランダムにカメラ数を選択
# seq_len_range: [4, 32]    # 各サンプルでランダムにシーケンス長を選択
```

### ロス設定

`src/plcs/configs/loss/multiview_sequence.yaml`:

```yaml
position_weight: 1.0
rotation_weight: 1.0
temporal:
  position_gt:
    weight: 0.1
    order: 2
    robust: true
  position_inertia:
    weight: 0.0
    order: 2
    robust: true
  rotation_gt:
    weight: 0.0
    order: 2
    robust: true
  rotation_inertia:
    weight: 0.0
    order: 2
    robust: true
```

### モデル設定

`src/plcs/configs/model/multiview.yaml`:

```yaml
hidden_dim: 256
num_layers: 4
num_heads: 8
dropout: 0.1
```

## 入出力形式

### 入力（バッチ）

マルチビューモデルはシーケンシャル入力に対応しています。

| フィールド | 形状 | 説明 |
|-----------|------|------|
| `human_kp` | `(B, N, T, 17, 2)` | 2D人物キーポイント（マルチカメラ×時系列） |
| `court_kp` | `(B, N, T, 20, 2)` | 2Dコートキーポイント |
| `human_vis` | `(B, N, T, 17)` | 人物KP可視性マスク |
| `court_vis` | `(B, N, T, 20)` | コートKP可視性マスク |
| `view_mask` | `(B, N)` | 有効カメラマスク（パディング用） |
| `seq_mask` | `(B, T)` | 有効フレームマスク（パディング用） |
| `num_views` | `(B,)` | 各サンプルの実際のカメラ数 |
| `seq_len` | `(B,)` | 各サンプルの実際のシーケンス長 |
| `position` | `(B, T, 3)` | 位置Ground Truth |
| `rotation` | `(B, T, 2)` | 回転Ground Truth（sin/cos） |

- `B`: バッチサイズ
- `N`: 最大カメラ数（バッチ内でパディング）
- `T`: 最大シーケンス長（バッチ内でパディング）

### 出力

| フィールド | 形状 | 説明 |
|-----------|------|------|
| `position` | `(B, T, 3)` | 推定3D位置（時系列） |
| `rotation` | `(B, T, 2)` | 推定回転（sin/cos、時系列） |

## 損失関数

- **位置損失**: SmoothL1Loss（Huber）
- **回転損失**: `1 - cosine_similarity`（sin/cos表現の単位ベクトル）
- **時間一貫性損失**: `temporal.*.weight > 0` の項目を加算（位置/回転 × GT/慣性）
- **総合損失**: `position_weight * pos_loss + rotation_weight * rot_loss + Σ temporal_term_weight * temporal_term_loss`

ロス設定は `src/plcs/configs/loss/multiview_sequence.yaml` で管理されます。

## 評価指標

- **Position Error (m)**: 3D位置のユークリッド距離誤差
- **Rotation Error (deg)**: 回転角度の誤差（度）

## 出力ファイル

学習完了後、`run.output_dir` に以下が生成されます：

```
outputs/plcs/multiview/
├── config.yaml          # 使用した設定のコピー
├── checkpoints/
│   ├── last.ckpt        # 最終チェックポイント
│   └── best.ckpt        # 最良チェックポイント
└── logs/
    └── tensorboard/     # TensorBoardログ
```

## 関連ファイル

- モデル: `src/plcs/models/plcs_multiview_model.py`
- データセット: `src/plcs/data/multiview_dataset.py`
- Lightning Module: `src/plcs/training/multiview_lightning_module.py`
- DataModule: `src/plcs/data/datamodule.py` (`PLCSMultiViewDataModule`)
