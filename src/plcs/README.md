# PLCS (Player Localization in Court System)

PLCS は、テニスコート座標系におけるプレイヤーの位置・向き（姿勢）を、
2D の姿勢観測（2D pose キーポイント）から推定するためのタスク実装です。

## 目的 / 想定入出力

- **入力**: 2D pose（関節座標）+ コートキーポイント
- **出力**: コート座標系でのプレイヤー 3D 位置・回転（6DoF）

### 入力形式

PLCSでは**camera-time順序**を採用しています: `(B, N, T, ...)` の順で、N=カメラ数、T=時系列長です。

| モード | 入力テンソル | 形状 | 説明 |
|--------|-------------|------|------|
| 単一カメラ (Frame) | `human_kp` | `(B, 17, 2)` | 2D人物キーポイント |
| 単一カメラ (Sequence) | `human_kp` | `(B, T, 17, 2)` | 時系列2D人物キーポイント |
| マルチビュー (Frame) | `human_kp` | `(B, N, 17, 2)` | 複数カメラからの2D人物キーポイント |
| マルチビュー (Sequence) | `human_kp` | `(B, N, T, 17, 2)` | 複数カメラ×時系列2D人物キーポイント (camera-time順) |

※ コートキーポイント `court_kp` も同様に各モードに対応

### 出力形式

| モード | position | rotation |
|--------|----------|----------|
| Frame | `(B, 3)` | `(B, 2)` |
| Sequence | `(B, T, 3)` | `(B, T, 2)` |

## ディレクトリ構成

```
src/plcs/
├── configs/                          # Hydra 設定ファイル群
│   ├── train.yaml                    # フレーム学習設定
│   ├── train_sequence.yaml           # シーケンス学習設定
│   ├── train_multiview.yaml          # マルチビュー学習設定
│   ├── generate_dataset.yaml         # データ生成設定
│   ├── visualize.yaml                # 単一カメラ可視化設定
│   ├── visualize_multiview.yaml      # マルチビュー可視化設定
│   ├── run/                          # 実行時設定（seed, gpus, output_dir）
│   ├── model/
│   │   ├── frame.yaml                # フレームモデル設定
│   │   ├── sequence.yaml             # シーケンスモデル設定
│   │   └── multiview.yaml            # マルチビューモデル設定
│   ├── data/
│   │   ├── frame.yaml                # フレーム DataModule 設定
│   │   ├── sequence.yaml             # シーケンス DataModule 設定
│   │   └── multiview.yaml            # マルチビュー DataModule 設定
│   ├── training/
│   │   └── default.yaml              # 学習ハイパーパラメータ
│   ├── loss/
│   │   ├── frame.yaml                # フレームロス設定（temporal=0）
│   │   ├── sequence.yaml             # シーケンスロス設定（temporal有効）
│   │   └── multiview_sequence.yaml   # マルチビューシーケンスロス設定
│   ├── simulation/
│   │   └── default.yaml              # シミュレーション設定（シーン数等）
│   ├── camera/
│   │   └── default.yaml              # カメラ投影パラメータ
│   └── visualization/
│       ├── default.yaml              # 単一カメラ可視化オプション
│       └── multiview.yaml            # マルチビュー可視化オプション
│
├── scripts/                          # 実行スクリプト（Hydra エントリポイント）
│   ├── generate_dataset.py           # SMPL-H モーションからの合成データ生成
│   ├── train.py                      # フレーム単位モデル学習
│   ├── train_sequence.py             # シーケンスモデル学習
│   ├── train_multiview.py            # マルチビューモデル学習
│   ├── visualize.py                  # 単一カメラ可視化
│   └── visualize_multiview.py        # マルチビュー可視化
│
├── models/                           # 推定モデル
│   ├── plcs_model.py                 # PLCSModel: フレーム単位 2D→3D 推定
│   ├── plcs_sequence_model.py        # PLCSSequenceModel: シーケンス対応
│   ├── plcs_multiview_model.py       # PLCSMultiViewModel: 複数視点統合
│   └── components/
│       ├── encoders.py               # 入力エンコーダ（キーポイント処理）
│       └── heads.py                  # 出力ヘッド（位置・回転回帰）
│
├── data/                             # データセット・DataModule
│   ├── dataset.py                    # フレーム Dataset
│   ├── sequence_dataset.py           # シーケンス Dataset
│   ├── multiview_dataset.py          # マルチビュー Dataset
│   └── datamodule.py                 # LightningDataModule（全モード共用）
│
├── training/                         # 学習関連
│   ├── lightning_module.py           # フレーム用 LightningModule
│   ├── sequence_lightning_module.py  # シーケンス用 LightningModule
│   ├── multiview_lightning_module.py # マルチビュー用 LightningModule
│   ├── losses.py                     # 損失関数（位置 MSE、回転損失）
│   └── metrics.py                    # 評価指標（位置誤差、角度誤差）
│
├── inference/                        # 推論
│   ├── predictor.py                  # フレーム推論
│   ├── sequence_predictor.py         # シーケンス推論
│   ├── multiview_predictor.py        # マルチビュー推論
│   └── visualization.py              # 3D プレーヤー描画
│
└── generate_dataset/                 # データセット生成ロジック
    ├── scene_generator.py            # シーン生成オーケストレータ
    ├── sampling/
    │   └── motion_sampler.py         # SMPL-H モーションのサンプリング
    └── io/
        └── dataset_io.py             # シーン保存・読込
```

## 主要コンポーネントの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ generate_dataset.py                                             │
│   ├── generate_dataset/scene_generator.py                       │
│   │   ├── sampling/motion_sampler.py  (SMPL-H モーション取得)    │
│   │   └── 複数カメラ投影 → 2D キーポイント生成                    │
│   └── → data/plcs/scenes/*.npz（各シーンに複数カメラ情報含む）    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ train.py / train_sequence.py / train_multiview.py               │
│   ├── data/datamodule.py           (DataModule)                 │
│   │   ├── PLCSDataModule           (単一カメラ・フレーム)         │
│   │   ├── PLCSSequenceDataModule   (単一カメラ・シーケンス)       │
│   │   └── PLCSMultiViewDataModule  (マルチビュー)                │
│   ├── models/                                                   │
│   │   ├── plcs_model.py            (フレームモデル)              │
│   │   ├── plcs_sequence_model.py   (シーケンスモデル)            │
│   │   └── plcs_multiview_model.py  (マルチビューモデル)          │
│   └── → outputs/plcs*/checkpoints/                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ visualize.py / visualize_multiview.py                           │
│   ├── inference/predictor.py            (単一カメラ推論)         │
│   ├── inference/multiview_predictor.py  (マルチビュー推論)       │
│   └── inference/visualization.py        (描画)                  │
└─────────────────────────────────────────────────────────────────┘
```

## 実行コマンド

詳細は [docs/scripts/plcs/](../../../docs/scripts/plcs/) を参照。

### データ生成

```bash
uv run python -m src.plcs.scripts.generate_dataset
```

### データ分布の調査（位置・向き・カメラ数）

生成済み PLCS データセット（`data/plcs/scenes/*.npz`）について、プレイヤー位置や yaw の分布、
原点近傍への偏り（半径しきい値内の割合）などを集計します。

```bash
uv run python -m src.plcs.scripts.analysis.analyze_dataset_distribution

# 出力先やサンプル数の変更例
uv run python -m src.plcs.scripts.analysis.analyze_dataset_distribution \
    run.output_dir=outputs/plcs_dist \
    analysis.max_scenes=200 \
    analysis.max_frames_per_scene=256
```

### 学習

```bash
# フレーム単位モデル（単一カメラ）
uv run python -m src.plcs.scripts.train

# シーケンスモデル（単一カメラ・時系列）
uv run python -m src.plcs.scripts.train_sequence

# マルチビューモデル（複数カメラ統合）
uv run python -m src.plcs.scripts.train_multiview

# マルチビュー学習のカスタム設定例
uv run python -m src.plcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2 \
    training.max_epochs=100
```

### 可視化

```bash
# 単一カメラ
uv run python -m src.plcs.scripts.visualize

# マルチビュー（Ground Truth）
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.scene_path=data/plcs/scenes/scene_000003.npz

# マルチビュー（チェックポイントからの予測）
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.checkpoint=outputs/plcs_multiview/checkpoints/last.ckpt \
    visualization.cameras=all

# 比較アニメーション出力
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.view=animation \
    visualization.save=comparison.mp4
```

## モデルアーキテクチャ

### フレームモデル (`PLCSModel`)

単一フレーム・単一カメラからの推定。MLP ベースのシンプルな構造。

### シーケンスモデル (`PLCSSequenceModel`)

時系列情報を活用した推定。LSTM または Transformer ベース。

### マルチビューモデル (`PLCSMultiViewModel`)

複数カメラからの観測を統合して推定。シーケンシャル入力 `(N_cam, T, ...)` に対応し、
時系列全体の位置・回転を推定します。

- **入力**: 複数カメラ×時系列のキーポイント `(B, N, T, 17, 2)`
- **出力**: 時系列の位置・回転 `(B, T, 3)`, `(B, T, 2)`
- **動的サンプリング**: `num_views_range`, `seq_len_range` で学習時に範囲からランダムにサンプリング可能
- **時間一貫性ロス**: 速度・加速度の平滑化による安定した軌道推定

現在の実装は view mean pooling によるスケルトン構造で、今後より高度な attention ベースの融合手法への拡張を予定しています。

## 設定ファイル

| 用途 | メイン設定 | 補助設定 |
|------|----------|---------|
| フレーム学習 | `train.yaml` | `model/frame.yaml`, `data/frame.yaml`, `loss/frame.yaml` |
| シーケンス学習 | `train_sequence.yaml` | `model/sequence.yaml`, `data/sequence.yaml`, `loss/sequence.yaml` |
| マルチビュー学習 | `train_multiview.yaml` | `model/multiview.yaml`, `data/multiview.yaml`, `loss/multiview_sequence.yaml` |
| 単一カメラ可視化 | `visualize.yaml` | `visualization/default.yaml` |
| マルチビュー可視化 | `visualize_multiview.yaml` | `visualization/multiview.yaml` |

### ロス設定ファイル

ロス関数のウェイトは `configs/loss/` ディレクトリで管理されています：

| ファイル | 用途 | temporal_weight |
|---------|------|-----------------|
| `frame.yaml` | フレーム単位学習 | 0.0 |
| `sequence.yaml` | シーケンス学習 | 0.1 |
| `multiview_sequence.yaml` | マルチビューシーケンス学習 | 0.1 |

詳細なドキュメントは以下を参照:
- [visualize_multiview.md](../../../docs/scripts/plcs/visualize_multiview.md) - マルチビュー可視化スクリプト
