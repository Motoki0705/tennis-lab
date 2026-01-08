# BLCS train_multiview.py

マルチビュー（複数カメラ）からのボール観測を使用して3D軌道を推定するモデルを学習するスクリプトです。

## 概要

`train_multiview.py` は、複数カメラからの2Dボール観測（画像座標シーケンス）とコートキーポイントを入力として、
コート座標系での3Dボール軌道を推定するモデルを学習します。

単一カメラの `train.py` とは異なり、複数視点の情報を融合することで、
深度推定の曖昧さを解消し、より正確な3D軌道復元を実現します。

## アーキテクチャ

PLCS multiview modelと同様の**Alternating Attention Architecture**を採用しています：

1. **BallCourtEncoder**: 各(フレーム, カメラ)ペアのボール位置とコートキーポイントをトークンにエンコード
2. **Alternating Attention Blocks**: Sequential AttentionとCamera Attentionを交互に適用
   - **Sequential Attention**: 各カメラについて時系列方向のアテンション
   - **Camera Attention**: 各フレームについてカメラ間のアテンション
3. **Aggregation MLP**: カメラトークンを集約してヘッドの入力次元に射影
4. **Trajectory3DHead**: 最終的な3D位置を出力

## 実行方法

### 基本実行

```bash
uv run python -m src.blcs.scripts.train_multiview
```

### 出力先・データセット指定

```bash
uv run python -m src.blcs.scripts.train_multiview \
    run.output_dir=outputs/blcs/multiview \
    data.scene_dir=data/blcs/scenes
```

### カメラ数の設定

```bash
# 4カメラ同時使用、最低2カメラ必須
uv run python -m src.blcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2
```

### Dry Run（データローディング確認のみ）

```bash
uv run python -m src.blcs.scripts.train_multiview run.dry_run=true
```

### GPU学習

```bash
uv run python -m src.blcs.scripts.train_multiview run.gpus=1
```

### 高速開発モード（1バッチのみ）

```bash
uv run python -m src.blcs.scripts.train_multiview run.fast_dev_run=true
```

## 設定ファイル構成

### メイン設定

`src/blcs/configs/train_multiview.yaml`:

```yaml
defaults:
  - data: multiview
  - model: multiview
  - training: default
  - run: train_multiview

run:
  seed: 42
  gpus: 0
  output_dir: outputs/blcs/multiview
  dry_run: false
  fast_dev_run: false

hydra:
  run:
    dir: ${run.output_dir}
```

### データ設定

`src/blcs/configs/data/multiview.yaml`:

```yaml
scene_dir: data/blcs/scenes
batch_size: 32
num_workers: 4
num_views: 3      # 同時に使用するカメラ数
min_cameras: 2    # シーンに必要な最小カメラ数
max_seq_len: 120  # 最大シーケンス長

# Range sampling for dynamic view/seq (optional)
# num_views_range: [1, 8]  # min, max inclusive
# seq_len_range: [15, 60]  # min, max inclusive
```

#### Range Sampling

学習時のデータ多様性を向上させるため、各サンプルごとにビュー数やシーケンス長をランダムにサンプリングできます：

```bash
# ビュー数を1〜4の範囲でランダムサンプリング
uv run python -m src.blcs.scripts.train_multiview \
    'data.num_views_range=[1, 4]'

# シーケンス長を15〜60の範囲でランダムサンプリング
uv run python -m src.blcs.scripts.train_multiview \
    'data.seq_len_range=[15, 60]'
```

### モデル設定

`src/blcs/configs/model/multiview.yaml`:

```yaml
hidden_dim: 256
num_layers: 4      # Alternating attention layer pairs
num_heads: 8
dropout: 0.1
max_seq_len: 120
max_views: 8
encoder_layers: 2  # Number of MLP layers in BallCourtEncoder
```

## 入出力形式

### 入力（バッチ）

モデルは `(B, N, T, ...)` 形式のテンソルを受け取ります：

| フィールド | 形状 | 説明 |
|-----------|------|------|
| `ball_uv` | `(B, N, T, 2)` | 2Dボール観測 |
| `court_kp` | `(B, N, T, 20, 2)` | 2Dコートキーポイント（各フレームに展開） |
| `ball_mask` | `(B, N, T)` | ボール可視性マスク |
| `court_vis` | `(B, N, T, 20)` | コートKP可視性マスク |
| `num_views` | `(B,)` | 有効カメラ数 |
| `seq_len` | `(B,)` | 実シーケンス長 |
| `position_3d` | `(B, T, 3)` | 軌道Ground Truth |

- `B`: バッチサイズ
- `N`: カメラ数（`num_views`）
- `T`: シーケンス長（パディング後は `max_seq_len`）

**Note**: コートキーポイントは従来の `(B, N, 20, 2)` 形式から `(B, T, N, 20, 2)` 形式に変更されました。
これにより、Alternating Attentionアーキテクチャで時系列方向のアテンションを効果的に適用できます。

### 出力

| フィールド | 形状 | 説明 |
|-----------|------|------|
| `position` | `(B, T, 3)` | 推定3D軌道 |

## 損失関数

- **位置損失**: MSE（平均二乗誤差）、有効フレームのみで計算

## 評価指標

- **Position Error (m)**: 各フレームでの3D位置ユークリッド距離誤差の平均

## 出力ファイル

学習完了後、`run.output_dir` に以下が生成されます：

```
outputs/blcs/multiview/
├── config.yaml          # 使用した設定のコピー
├── checkpoints/
│   ├── last.ckpt        # 最終チェックポイント
│   └── best.ckpt        # 最良チェックポイント
└── logs/
    └── tensorboard/     # TensorBoardログ
```

## 関連ファイル

- モデル: `src/blcs/models/blcs_multiview_model.py`
- エンコーダ: `src/blcs/models/components/encoders.py` (`BallCourtEncoder`)
- データセット: `src/blcs/data/multiview_dataset.py`
- Lightning Module: `src/blcs/training/multiview_lightning_module.py`
- DataModule: `src/blcs/data/datamodule.py` (`BLCSMultiViewDataModule`)
