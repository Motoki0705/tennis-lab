# PLCS (Player Localization in Court System)

PLCS は、テニスコート座標系におけるプレイヤーの位置・向き（姿勢）を、2D の姿勢観測（2D pose キーポイント）から推定するためのタスク実装です。

## 目的 / 想定入出力

- **入力**: 2D pose（関節座標）+ コートキーポイント
- **出力**: コート座標系でのプレイヤー 3D 位置・回転（6DoF）

## ディレクトリ構成

```
src/plcs/
├── configs/                          # Hydra 設定ファイル群
│   ├── train.yaml                    # フレーム学習メイン設定
│   ├── train_sequence.yaml           # シーケンス学習メイン設定
│   ├── generate_dataset.yaml         # データ生成メイン設定
│   ├── visualize.yaml                # 可視化メイン設定
│   ├── run/                          # 実行時設定（seed, gpus, output_dir 等）
│   │   ├── train.yaml
│   │   ├── train_sequence.yaml
│   │   └── generate_dataset.yaml
│   ├── model/
│   │   ├── frame.yaml                # フレームモデル設定
│   │   └── sequence.yaml             # シーケンスモデル設定（LSTM/Transformer）
│   ├── data/
│   │   ├── frame.yaml                # フレーム DataModule 設定
│   │   └── sequence.yaml             # シーケンス DataModule 設定
│   ├── training/
│   │   └── default.yaml              # 学習ハイパーパラメータ
│   ├── metrics/
│   │   └── default.yaml              # 評価指標設定
│   ├── simulation/
│   │   └── default.yaml              # シミュレーション設定（シーン数等）
│   ├── camera/
│   │   └── default.yaml              # カメラ投影パラメータ
│   ├── paths/
│   │   └── default.yaml              # データパス設定
│   ├── motion_sources/
│   │   └── default.yaml              # SMPL-H モーションデータ設定
│   └── visualization/
│       └── default.yaml              # 可視化オプション
│
├── scripts/                          # 実行スクリプト（Hydra エントリポイント）
│   ├── generate_dataset.py           # SMPL-H モーションからの合成データ生成
│   ├── train.py                      # フレーム単位モデル学習
│   ├── train_sequence.py             # シーケンスモデル学習
│   └── visualize.py                  # シーン・予測結果の可視化
│
├── generate_dataset/                 # データセット生成ロジック
│   ├── scene_generator.py            # シーン生成オーケストレータ（SMPL-H→投影→シーン）
│   ├── sampling/
│   │   └── motion_sampler.py         # SMPL-H モーションのサンプリング
│   └── io/
│       └── dataset_io.py             # シーン保存・読込
│
├── models/                           # 推定モデル
│   ├── plcs_model.py                 # PLCSModel: フレーム単位 2D→3D 推定
│   ├── plcs_sequence_model.py        # PLCSSequenceModel: シーケンス対応
│   └── components/
│       ├── encoders.py               # 入力エンコーダ（キーポイント処理）
│       └── heads.py                  # 出力ヘッド（位置・回転回帰）
│
├── data/                             # データセット・DataModule
│   ├── dataset.py                    # フレーム Dataset
│   ├── sequence_dataset.py           # シーケンス Dataset
│   └── datamodule.py                 # LightningDataModule（frame/sequence 共用）
│
├── training/                         # 学習関連
│   ├── lightning_module.py           # フレーム用 LightningModule
│   ├── sequence_lightning_module.py  # シーケンス用 LightningModule
│   ├── losses.py                     # 損失関数（位置 MSE、回転損失）
│   └── metrics.py                    # 評価指標（位置誤差、角度誤差）
│
└── inference/                        # 推論・可視化
    ├── predictor.py                  # フレーム推論
    ├── sequence_predictor.py         # シーケンス推論
    └── visualization.py              # 3D プレーヤー描画
```

## 主要コンポーネントの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ generate_dataset.py                                             │
│   ├── generate_dataset/scene_generator.py                       │
│   │   ├── sampling/motion_sampler.py  (SMPL-H モーション取得)    │
│   │   └── カメラ投影 → 2D キーポイント生成                        │
│   └── → data/plcs/scenes/*.npz                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ train.py / train_sequence.py                                    │
│   ├── data/datamodule.py           (DataModule)                 │
│   ├── models/plcs_model.py         (フレームモデル)              │
│   ├── models/plcs_sequence_model.py(シーケンスモデル)            │
│   ├── training/*_lightning_module.py                            │
│   └── → outputs/plcs/checkpoints/                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ visualize.py                                                    │
│   ├── inference/predictor.py       (推論)                       │
│   └── inference/visualization.py   (描画)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 実行コマンド

詳細は [docs/scripts/plcs/](../../../docs/scripts/plcs/) を参照。

```bash
# データ生成
uv run python -m src.plcs.scripts.generate_dataset

# 学習（フレーム）
uv run python -m src.plcs.scripts.train

# 学習（シーケンス）
uv run python -m src.plcs.scripts.train_sequence

# 可視化
uv run python -m src.plcs.scripts.visualize
```

## マルチビュー推定（Multi-View Inference）

複数カメラからの観測を統合して、より高精度にプレイヤーの位置・回転を推定する機能です。

### 概要

単一カメラでは遮蔽や視野角の制約により推定精度が制限される場合がありますが、
マルチビューでは複数視点からの情報を融合することで、より堅牢な推定が可能になります。

### 主要コンポーネント

- **MultiViewSceneDataset** (`data/multiview_dataset.py`):
  複数カメラの観測をまとめて返す Dataset。`num_views` パラメータで同時に使用するカメラ数を指定。

- **PLCSMultiViewModel** (`models/plcs_multiview_model.py`):
  複数視点からの入力を受け取り、統合した上で3D位置・回転を出力するモデル。
  ※現在はスケルトン実装（view mean pooling）のみ。アーキテクチャは今後拡張予定。

- **PLCSMultiViewLightningModule** (`training/multiview_lightning_module.py`):
  マルチビューモデル用の Lightning モジュール。

- **PLCSMultiViewDataModule** (`data/datamodule.py`):
  マルチビュー Dataset を管理する DataModule。

### 入出力形式

**入力**:
- `human_kp`: `(B, N, 17, 2)` - 各カメラからの2D人物キーポイント（B: バッチ, N: カメラ数）
- `court_kp`: `(B, N, 20, 2)` - 各カメラからの2Dコートキーポイント
- `human_kp_mask`: `(B, N, 17)` - 人物キーポイント可視性マスク
- `court_kp_mask`: `(B, N, 20)` - コートキーポイント可視性マスク
- `view_mask`: `(B, N)` - 有効なカメラのマスク（パディング用）

**出力**:
- `position`: `(B, 3)` - 3D位置（コート座標系）
- `rotation`: `(B, 2)` - 回転（sin/cos）

### 実行コマンド

```bash
# マルチビュー学習
uv run python -m src.plcs.scripts.train_multiview

# カスタム設定
uv run python -m src.plcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2 \
    training.max_epochs=100

# Dry Run（データローディングのみ確認）
uv run python -m src.plcs.scripts.train_multiview run.dry_run=true
```

### 設定ファイル

- `configs/train_multiview.yaml`: マルチビュー学習メイン設定
- `configs/data/multiview.yaml`: マルチビュー DataModule 設定
- `configs/model/multiview.yaml`: マルチビューモデル設定
