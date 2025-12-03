# Ball Localization in Court System (BLCS)

BLCSは、2Dボール軌道とコートキーポイントから、テニスコート座標系における3Dボール軌道を推定するモデルです。

## 概要

### 入力（1シーン・1カメラ）
- **ボールの2D軌道** `[T, 2]`: 各フレームのボール中心位置 (u, v) - 画像サイズで正規化
- **コートの2Dキーポイント** `[20, 2]`: CourtKP20の投影座標

### 出力
- **3D軌道** `[T, 3]`: コート座標系でのボール位置 (x, y, z) - 正規化座標

## インストール

```bash
cd tennis-lab
pip install -e .
```

## 使用方法

### 1. データセット生成

```bash
# YAML設定ファイルを使用（推奨）
python -m blcs.scripts.generate_dataset \
    --config blcs/configs/dataset.yaml \
    --output-dir data/blcs

# コマンドライン引数で指定
python -m blcs.scripts.generate_dataset \
    --output-dir data/blcs \
    --samples-per-cell 100 \
    --num-cameras-sampled 8 \
    --ball-visibility-threshold 0.8
```

### 2. データセットの可視化

```bash
# シーン情報を表示
python -m blcs.scripts.visualize_scene data/blcs/scenes/scene_000000.npz --info

# マルチビュー表示（3D + 2D + カメラビュー）
python -m blcs.scripts.visualize_scene data/blcs/scenes/scene_000000.npz --view multi

# アニメーション
python -m blcs.scripts.visualize_scene data/blcs/scenes/scene_000000.npz --view animation

# ファイルに保存
python -m blcs.scripts.visualize_scene data/blcs/scenes/scene_000000.npz --save output.png
```

### 3. 学習

```bash
python -m blcs.scripts.train \
    --config blcs/configs/default.yaml \
    --gpus 1 \
    --output-dir outputs/blcs
```

### 4. 推論

```bash
python -m blcs.scripts.predict \
    --checkpoint outputs/blcs/checkpoints/last.ckpt \
    --input data/sample.pt \
    --visualize
```

### 5. Pythonからの使用

```python
from blcs.inference.predictor import BLCSPredictor

# モデルをロード
predictor = BLCSPredictor(
    checkpoint_path="outputs/blcs/checkpoints/last.ckpt",
    device="cuda",
)

# 予測
outputs = predictor.predict(
    ball_uv=ball_trajectory,      # [T, 2]
    court_kp=court_keypoints,     # [20, 2]
    denormalize=True,             # メートル単位で出力
)

trajectory_3d = outputs["position"]  # [T, 3] in meters
```

## ディレクトリ構成

```
blcs/
├── configs/           # 設定ファイル
│   ├── default.yaml   # 学習設定
│   └── dataset.yaml   # データセット生成設定
├── data/              # データパイプライン
│   ├── dataset.py     # PyTorch Dataset
│   ├── datamodule.py  # Lightning DataModule
│   ├── scene_generator.py    # シーン生成
│   ├── dataset_writer.py     # NPZ保存
│   ├── camera_projector.py   # カメラ投影
│   └── distribution_sampler.py  # 分布制御
├── inference/         # 推論モジュール
├── models/            # モデル定義
├── scripts/           # CLIスクリプト
│   ├── generate_dataset.py  # データセット生成
│   ├── visualize_scene.py   # 可視化
│   └── train.py             # 学習
├── simulation/        # 物理シミュレーション
│   ├── ball_physics.py      # ボール物理
│   ├── shot_simulator.py    # ショット生成
│   └── cell_manager.py      # セル分割
├── training/          # 学習モジュール
└── utils/             # ユーティリティ
```

## データセット形式（PLCS統一形式）

1シーン = 1ファイル（複数カメラ）:

```
scene_000000.npz
├── meta (JSON)           # シーンメタデータ
├── ball_pos_world [T, 3] # ワールド座標（メートル）
├── ball_pos_norm [T, 3]  # 正規化座標（学習ターゲット）
├── ball_vel_world [T, 3] # 速度
├── num_cameras           # カメラ数
├── cam_0_params (JSON)   # カメラ0のパラメータ
├── cam_0_ball_uv [T, 2]  # カメラ0でのボールUV
├── cam_0_ball_visible [T]# カメラ0での可視性
├── cam_0_court_kp_uv [20, 2]     # カメラ0でのコートKP
├── cam_0_court_kp_visible [20]   # カメラ0でのKP可視性
├── cam_0_ball_visibility_ratio   # ボール視認率
├── cam_1_...             # カメラ1のデータ
└── ...
```

## アーキテクチャ

```
Input: ball_uv [B, T, 2], court_kp [B, 20, 2]
         │                      │
         ▼                      ▼
  BallTrajectoryEncoder   CourtContextEncoder
         │                      │
         ▼                      ▼
    [B, T, D]              [B, D]
         │                      │
         └──────┬───────────────┘
                │
                ▼
        CourtBallCrossAttention
                │
                ▼
           [B, T, D]
                │
                ▼
         Trajectory3DHead
                │
                ▼
    Output: position [B, T, 3]
```

## 座標系

### 入力UV座標
- `u = x / W` (画像幅で正規化, 0〜1)
- `v = y / H` (画像高さで正規化, 0〜1)

### 出力コート座標（正規化）
- `x = X / HALF_DOUBLES_WIDTH` (5.485m)
- `y = Y / HALF_LENGTH` (11.885m)
- `z = Z / NET_HEIGHT_POST` (1.07m)

## 評価指標

- **mean_position_error_m**: 平均位置誤差（メートル）
- **position_accuracy_0_3m**: GT から 0.3m 以内のフレーム割合（`position_threshold_m` の 1倍）
- **position_accuracy_0_6m**: GT から 0.6m 以内のフレーム割合（`position_threshold_m` の 2倍）
- **position_accuracy_1_2m**: GT から 1.2m 以内のフレーム割合（`position_threshold_m` の 4倍）
- **endpoint_accuracy_0_5m**: 終端位置が GT から 0.5m 以内のシーン割合（`endpoint_threshold_m` の 1倍）
- **endpoint_accuracy_1m**: 終端位置が GT から 1.0m 以内のシーン割合（`endpoint_threshold_m` の 2倍）

## PLCSとの共通モジュール

以下のモジュールは `plcs/` から再利用しています：

- `plcs.utils.court`: コートジオメトリ、カメラ投影
- `plcs.utils.constants`: コートキーポイント定義
- `plcs.rendering`: コート描画（オプション）
