# BLCS (Ball Localization in Court System)

BLCS は、テニスコート座標系におけるボールの 3D 軌道を、2D のボール観測（画像座標）と
コート情報から推定するためのタスク実装です。

## 目的 / 想定入出力

- **入力**: 2D ボール観測（画像座標シーケンス）+ コートキーポイント
- **出力**: コート座標系の 3D 位置（軌道）

### 入力形式

| モード | 入力テンソル | 形状 | 説明 |
|--------|-------------|------|------|
| 単一カメラ | `ball_uv` | `(B, T, 2)` | 2Dボール観測シーケンス |
| マルチビュー | `ball_uv` | `(B, N, T, 2)` | 複数カメラからの2Dボール観測 |

※ コートキーポイント `court_kp` も同様に各モードに対応

### 出力形式

- `position`: `(B, T, 3)` - 3D軌道（コート座標系）

## ディレクトリ構成

```
src/blcs/
├── configs/                          # Hydra 設定ファイル群
│   ├── train.yaml                    # 単一カメラ学習設定
│   ├── train_multiview.yaml          # マルチビュー学習設定
│   ├── generate_dataset.yaml         # データ生成設定
│   ├── visualize.yaml                # 単一カメラ可視化設定
│   ├── visualize_multiview.yaml      # マルチビュー可視化設定
│   ├── run/                          # 実行時設定（seed, gpus, output_dir）
│   ├── model/
│   │   ├── default.yaml              # 単一カメラモデル設定
│   │   └── multiview.yaml            # マルチビューモデル設定
│   ├── data/
│   │   ├── default.yaml              # 単一カメラ DataModule 設定
│   │   └── multiview.yaml            # マルチビュー DataModule 設定
│   ├── training/
│   │   └── default.yaml              # 学習ハイパーパラメータ
│   ├── physics/
│   │   └── default.yaml              # ボール物理パラメータ
│   ├── shot/
│   │   └── default.yaml              # ショット種別・初速設定
│   ├── camera/
│   │   └── default.yaml              # カメラ投影パラメータ
│   └── visualization/
│       ├── default.yaml              # 単一カメラ可視化オプション
│       └── multiview.yaml            # マルチビュー可視化オプション
│
├── scripts/                          # 実行スクリプト（Hydra エントリポイント）
│   ├── generate_dataset.py           # 物理シミュレーションによるデータ生成
│   ├── train.py                      # 単一カメラモデル学習
│   ├── train_multiview.py            # マルチビューモデル学習
│   ├── run_hparam_sweep.sh           # 単一カメラモデルのハイパラ探索
│   ├── visualize.py                  # 単一カメラ可視化
│   └── visualize_multiview.py        # マルチビュー可視化
│
├── simulation/                       # 物理シミュレーション
│   ├── ball_physics.py               # ボール運動の物理計算
│   ├── shot_simulator.py             # ショットシミュレータ（軌道生成）
│   └── cell_manager.py               # コートセル分割・打点管理
│
├── models/                           # 推定モデル
│   ├── blcs_model.py                 # BLCSModel: 単一カメラ 2D→3D 軌道推定
│   ├── blcs_multiview_model.py       # BLCSMultiViewModel: 複数視点統合
│   └── components/
│       ├── encoders.py               # 入力エンコーダ
│       └── heads.py                  # 出力ヘッド（3D 座標回帰）
│
├── data/                             # データセット・DataModule
│   ├── dataset.py                    # 単一カメラ Dataset
│   ├── multiview_dataset.py          # マルチビュー Dataset
│   └── datamodule.py                 # LightningDataModule（全モード共用）
│
├── training/                         # 学習関連
│   ├── lightning_module.py           # 単一カメラ用 LightningModule
│   ├── multiview_lightning_module.py # マルチビュー用 LightningModule
│   ├── losses.py                     # 損失関数（MSE、軌道損失等）
│   └── metrics.py                    # 評価指標（3D 位置誤差等）
│
├── inference/                        # 推論
│   ├── predictor.py                  # 単一カメラ推論
│   ├── multiview_predictor.py        # マルチビュー推論
│   └── visualization.py              # 3D 軌道の描画
│
├── generate_dataset/                 # データセット生成ロジック
│   ├── scene_generator.py            # シーン生成オーケストレータ
│   ├── sampling/
│   │   └── distribution_sampler.py   # 打点・方向の確率分布サンプリング
│   └── io/
│       └── dataset_io.py             # シーン保存・読込
│
└── demo/                             # デモ・プロトタイプ
    ├── app.py                        # Gradio/Streamlit デモアプリ
    ├── pipeline.py                   # デモ用パイプライン
    └── ...
```

## 主要コンポーネントの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ generate_dataset.py                                             │
│   ├── simulation/shot_simulator.py  (物理軌道生成)               │
│   ├── simulation/ball_physics.py    (物理計算)                  │
│   ├── 複数カメラへの投影 → 2D ボール座標生成                      │
│   └── → data/blcs/scenes/*.npz（各シーンに複数カメラ情報含む）    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ train.py / train_multiview.py                                   │
│   ├── data/datamodule.py           (DataModule)                 │
│   │   ├── BLCSDataModule           (単一カメラ)                  │
│   │   └── BLCSMultiViewDataModule  (マルチビュー)                │
│   ├── models/                                                   │
│   │   ├── blcs_model.py            (単一カメラモデル)            │
│   │   └── blcs_multiview_model.py  (マルチビューモデル)          │
│   └── → outputs/blcs/*/logs/version_*/checkpoints/              │
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

詳細は [docs/scripts/blcs/](../../../docs/scripts/blcs/) を参照。

### データ生成

```bash
uv run python -m src.blcs.scripts.generate_dataset
```

### 学習

```bash
# 単一カメラモデル
uv run python -m src.blcs.scripts.train

# マルチビューモデル（複数カメラ統合）
uv run python -m src.blcs.scripts.train_multiview

# マルチビュー学習のカスタム設定例
uv run python -m src.blcs.scripts.train_multiview \
    data.num_views=4 \
    data.min_cameras=2 \
    training.max_epochs=100
```

### ハイパーパラメータ探索

```bash
UV_CACHE_DIR=agents_workspace/tmp_cache/uv_cache \
RUN_ROOT=outputs/blcs/sweep_$(date +%Y-%m-%d_%H-%M-%S) \
bash src/blcs/scripts/run_hparam_sweep.sh
```

## データローディング最適化

- `data.cache_max_scenes`: シーンNPZのLRUキャッシュ容量。
- `data.scene_sampler_mode`: `none | scene | mixed | chunked` を指定。
- `data.chunk_max_scenes`: `chunked` モード時のチャンク内シーン数上限。

### 可視化

```bash
# 単一カメラ
uv run python -m src.blcs.scripts.visualize

# マルチビュー（Ground Truth）
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.scene_path=data/blcs/scenes/scene_000003.npz

# マルチビュー（チェックポイントからの予測）
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.checkpoint=outputs/blcs/multiview/logs/version_0/checkpoints/last.ckpt \
    visualization.cameras=all

# 予測結果をファイルに出力
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.output=predictions.json

# 比較アニメーション出力
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.view=animation \
    visualization.save=comparison.mp4
```

## モデルアーキテクチャ

### 単一カメラモデル (`BLCSModel`)

単一カメラからの2D観測シーケンスを入力として3D軌道を推定。
Temporal Encoder + MLP ベースの構造。

### マルチビューモデル (`BLCSMultiViewModel`)

複数カメラからの観測を統合して推定。単一カメラではボールの深度情報の推定が困難ですが、マルチビューでは複数視点からの2D観測を三角測量的に融合することで、より正確な3D軌道復元が可能です。

現在の実装は view pooling + temporal transformer によるスケルトン構造で、今後より高度な融合手法への拡張を予定しています。

## 設定ファイル

| 用途 | メイン設定 | 補助設定 |
|------|----------|---------|
| 単一カメラ学習 | `train.yaml` | `model/default.yaml`, `data/default.yaml` |
| マルチビュー学習 | `train_multiview.yaml` | `model/multiview.yaml`, `data/multiview.yaml` |
| 単一カメラ可視化 | `visualize.yaml` | `visualization/default.yaml` |
| マルチビュー可視化 | `visualize_multiview.yaml` | `visualization/multiview.yaml` |

詳細なドキュメントは以下を参照:
- [visualize_multiview.md](../../../docs/scripts/blcs/visualize_multiview.md) - マルチビュー可視化スクリプト
