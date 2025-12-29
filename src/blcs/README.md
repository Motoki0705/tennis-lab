# BLCS (Ball Localization in Court System)

BLCS は、テニスコート座標系におけるボールの 3D 軌道を、2D のボール観測（画像座標）とコート情報から推定するためのタスク実装です。

## 目的 / 想定入出力

- **入力**: 2D ボール観測（画像座標シーケンス）+ コートキーポイント
- **出力**: コート座標系の 3D 位置（軌道）

## ディレクトリ構成

```
src/blcs/
├── configs/                          # Hydra 設定ファイル群
│   ├── train.yaml                    # 学習メイン設定
│   ├── generate_dataset.yaml         # データ生成メイン設定
│   ├── visualize.yaml                # 可視化メイン設定
│   ├── run/                          # 実行時設定（seed, gpus, output_dir 等）
│   │   ├── train.yaml
│   │   ├── generate_dataset.yaml
│   │   └── visualize.yaml
│   ├── model/
│   │   └── default.yaml              # BLCSModel アーキテクチャ設定
│   ├── data/
│   │   └── default.yaml              # DataModule 設定
│   ├── training/
│   │   └── default.yaml              # 学習ハイパーパラメータ
│   ├── metrics/
│   │   └── default.yaml              # 評価指標設定
│   ├── physics/
│   │   └── default.yaml              # ボール物理パラメータ（重力、反発係数等）
│   ├── shot/
│   │   └── default.yaml              # ショット種別・初速設定
│   ├── sampling/
│   │   └── default.yaml              # サンプリング分布設定
│   ├── camera/
│   │   └── default.yaml              # カメラ投影パラメータ
│   ├── generator/
│   │   └── default.yaml              # シーン生成設定
│   └── visualization/
│       └── default.yaml              # 可視化オプション
│
├── scripts/                          # 実行スクリプト（Hydra エントリポイント）
│   ├── generate_dataset.py           # 物理シミュレーションによるデータ生成
│   ├── train.py                      # モデル学習
│   └── visualize.py                  # シーン・予測結果の可視化
│
├── simulation/                       # 物理シミュレーション
│   ├── ball_physics.py               # ボール運動の物理計算（放物線、バウンス、空気抵抗）
│   ├── shot_simulator.py             # ショットシミュレータ（軌道生成）
│   └── cell_manager.py               # コートセル分割・打点管理
│
├── generate_dataset/                 # データセット生成ロジック
│   ├── scene_generator.py            # シーン生成オーケストレータ
│   ├── sampling/
│   │   └── distribution_sampler.py   # 打点・方向の確率分布サンプリング
│   └── io/
│       └── dataset_io.py             # シーン保存・読込
│
├── models/                           # 推定モデル
│   ├── blcs_model.py                 # BLCSModel: 2D→3D 軌道推定ネットワーク
│   └── components/
│       ├── encoders.py               # 入力エンコーダ（MLP/Conv ベース）
│       └── heads.py                  # 出力ヘッド（3D 座標回帰）
│
├── data/                             # データセット・DataModule
│   ├── dataset.py                    # PyTorch Dataset（シーンファイル読込）
│   └── datamodule.py                 # LightningDataModule
│
├── training/                         # 学習関連
│   ├── lightning_module.py           # LightningModule（学習・評価ステップ）
│   ├── losses.py                     # 損失関数（MSE、軌道損失等）
│   └── metrics.py                    # 評価指標（3D 位置誤差等）
│
├── inference/                        # 推論・可視化
│   ├── predictor.py                  # チェックポイントからの推論
│   └── visualization.py              # 3D 軌道の描画
│
└── demo/                             # デモ・プロトタイプ
    ├── app.py                        # Gradio/Streamlit デモアプリ
    ├── pipeline.py                   # デモ用パイプライン
    ├── video_processor.py            # 動画処理
    ├── court_annotator.py            # コートアノテーション
    ├── simple_demo.py                # シンプルデモ
    └── test_annotator.py             # アノテータテスト
```

## 主要コンポーネントの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ generate_dataset.py                                             │
│   ├── simulation/shot_simulator.py  (物理軌道生成)               │
│   ├── simulation/ball_physics.py    (物理計算)                  │
│   ├── generate_dataset/scene_generator.py                       │
│   └── → data/blcs/scenes/*.npz                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ train.py                                                        │
│   ├── data/datamodule.py           (DataModule)                 │
│   ├── models/blcs_model.py         (ネットワーク)                │
│   ├── training/lightning_module.py (学習ループ)                  │
│   └── → outputs/blcs/checkpoints/                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ visualize.py                                                    │
│   ├── inference/predictor.py       (推論)                       │
│   └── inference/visualization.py   (描画)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 実行コマンド

詳細は [docs/scripts/blcs/](../../../docs/scripts/blcs/) を参照。

```bash
# データ生成
uv run python -m src.blcs.scripts.generate_dataset

# 学習
uv run python -m src.blcs.scripts.train

# 可視化
uv run python -m src.blcs.scripts.visualize
```

## マルチビュー推定（Multi-View Inference）

複数カメラからのボール観測を統合して、より高精度に3D軌道を推定する機能です。

### 概要

単一カメラではボールの深度情報の推定が困難ですが、
マルチビューでは複数視点からの2D観測を三角測量的に融合することで、
より正確な3D軌道復元が可能になります。

### 主要コンポーネント

- **MultiViewBallTrajectoryDataset** (`data/multiview_dataset.py`):
  複数カメラのボール観測をまとめて返す Dataset。`num_views` パラメータで同時に使用するカメラ数を指定。

- **BLCSMultiViewModel** (`models/blcs_multiview_model.py`):
  複数視点からの入力を受け取り、統合した上で3D軌道を出力するモデル。
  ※現在はスケルトン実装（view pooling + temporal transformer）のみ。アーキテクチャは今後拡張予定。

- **BLCSMultiViewLightningModule** (`training/multiview_lightning_module.py`):
  マルチビューモデル用の Lightning モジュール。

- **BLCSMultiViewDataModule** (`data/datamodule.py`):
  マルチビュー Dataset を管理する DataModule。

### 入出力形式

**入力**:
- `ball_uv`: `(B, N, T, 2)` - 各カメラからの2Dボール観測（B: バッチ, N: カメラ数, T: フレーム数）
- `court_kp`: `(B, N, 20, 2)` - 各カメラからの2Dコートキーポイント
- `ball_mask`: `(B, N, T)` - ボール可視性マスク
- `court_kp_mask`: `(B, N, 20)` - コートキーポイント可視性マスク
- `view_mask`: `(B, N)` - 有効なカメラのマスク（パディング用）
- `seq_len`: `(B,)` - 各サンプルの実際のシーケンス長

**出力**:
- `position`: `(B, T, 3)` - 3D軌道（コート座標系）

### 実行コマンド

```bash
# マルチビュー学習
uv run python -m src.blcs.scripts.train_multiview

# カスタム設定
uv run python -m src.blcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2 \
    training.max_epochs=100

# Dry Run（データローディングのみ確認）
uv run python -m src.blcs.scripts.train_multiview run.dry_run=true
```

### 設定ファイル

- `configs/train_multiview.yaml`: マルチビュー学習メイン設定
- `configs/data/multiview.yaml`: マルチビュー DataModule 設定
- `configs/model/multiview.yaml`: マルチビューモデル設定
