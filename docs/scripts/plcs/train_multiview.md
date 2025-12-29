# PLCS train_multiview.py

マルチビュー（複数カメラ）からの観測を使用してプレイヤーの3D位置・回転を推定するモデルを学習するスクリプトです。

## 概要

`train_multiview.py` は、複数カメラからの2D人物キーポイントとコートキーポイントを入力として、
コート座標系でのプレイヤーの3D位置と回転（向き）を推定するモデルを学習します。

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
    run.output_dir=outputs/plcs_multiview \
    data.scene_dir=data/plcs/scenes
```

### カメラ数の設定

```bash
# 4カメラ同時使用、最低2カメラ必須
uv run python -m src.plcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2
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
  output_dir: outputs/plcs_multiview
  dry_run: false
  fast_dev_run: false

hydra:
  run:
    dir: ${run.output_dir}
```

### データ設定

`src/plcs/configs/data/multiview.yaml`:

```yaml
scene_dir: data/plcs/scenes
batch_size: 32
num_workers: 4
num_views: 3      # 同時に使用するカメラ数
min_cameras: 2    # シーンに必要な最小カメラ数
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

| フィールド | 形状 | 説明 |
|-----------|------|------|
| `human_kp` | `(B, N, 17, 2)` | 2D人物キーポイント |
| `court_kp` | `(B, N, 20, 2)` | 2Dコートキーポイント |
| `human_kp_mask` | `(B, N, 17)` | 人物KP可視性マスク |
| `court_kp_mask` | `(B, N, 20)` | コートKP可視性マスク |
| `view_mask` | `(B, N)` | 有効カメラマスク |
| `position_gt` | `(B, 3)` | 位置Ground Truth |
| `rotation_gt` | `(B, 2)` | 回転Ground Truth（sin/cos） |

- `B`: バッチサイズ
- `N`: カメラ数（`num_views`）

### 出力

| フィールド | 形状 | 説明 |
|-----------|------|------|
| `position` | `(B, 3)` | 推定3D位置 |
| `rotation` | `(B, 2)` | 推定回転（sin/cos） |

## 損失関数

- **位置損失**: MSE（平均二乗誤差）
- **回転損失**: MSE（sin/cos表現）
- **総合損失**: `position_loss + rotation_weight * rotation_loss`

## 評価指標

- **Position Error (m)**: 3D位置のユークリッド距離誤差
- **Rotation Error (deg)**: 回転角度の誤差（度）

## 出力ファイル

学習完了後、`run.output_dir` に以下が生成されます：

```
outputs/plcs_multiview/
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
