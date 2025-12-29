# BLCS generate_dataset

画面上のシーケンシャルなボールの位置とコートのキーポイントを用いて3D上のボールの軌道を推定するための合成データセットを生成するスクリプト。

## 概要

このスクリプトは、テニスボールの物理シミュレーション（重力、空気抵抗、マグナス効果、バウンス）を使用して、様々なショットパターンのボール軌道データを生成します。生成されたデータは、2Dボール位置シーケンスから3D軌道を推定するモデルの学習に使用されます。

## コマンド例

```bash
# デフォルト設定で実行
uv run python -m src.blcs.scripts.generate_dataset

# 出力先とサンプル数を指定
uv run python -m src.blcs.scripts.generate_dataset run.output_dir=data/blcs sampling.per_from_cell_samples=10

# シード値を変更
uv run python -m src.blcs.scripts.generate_dataset run.seed=123

# データ分割比率を変更
uv run python -m src.blcs.scripts.generate_dataset run.train_ratio=0.7 run.val_ratio=0.15
```

## コンフィグ

エントリポイント: `src/blcs/configs/generate_dataset.yaml`

### defaults 構成

```yaml
defaults:
  - physics: default
  - shot: default
  - camera: default
  - sampling: default
  - generator: default
  - run: generate_dataset
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `data/blcs` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `device` | `cpu` | デバイス |
| `train_ratio` | `0.8` | 学習データの割合 |
| `val_ratio` | `0.1` | 検証データの割合 |

### physics (物理シミュレーション設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `gravity` | `9.81` | 重力加速度 (m/s²) |
| `k_drag` | `0.01` | 空気抵抗係数 |
| `k_magnus` | `0.001` | マグナス効果係数 |
| `e_z` | `0.75` | 垂直反発係数 |
| `mu` | `0.1` | 摩擦係数 |
| `alpha_net` | `0.3` | ネット衝突減衰係数 |
| `dt` | `0.00416667` | シミュレーション時間刻み (1/240秒) |
| `use_drag` | `true` | 空気抵抗を使用するか |
| `use_magnus` | `true` | マグナス効果を使用するか |

### shot (ショット設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `z_range` | `[0.8, 1.4]` | 発射点の高さ範囲 (m) |
| `speed_range` | `[15.0, 35.0]` | 初速範囲 (m/s) |
| `azimuth_range_deg` | `[-30.0, 30.0]` | 水平角度範囲 (度) |
| `elevation_range_deg` | `[5.0, 25.0]` | 仰角範囲 (度) |
| `spin_x_range` | `[-20.0, 20.0]` | X軸スピン範囲 (rad/s) |
| `spin_y_range` | `[-80.0, -40.0]` | Y軸スピン範囲 (トップスピン) |
| `spin_z_range` | `[-20.0, 20.0]` | Z軸スピン範囲 |
| `max_sim_frames` | `2000` | 最大シミュレーションフレーム数 |
| `output_fps` | `30` | 出力FPS |
| `sim_fps` | `240` | シミュレーションFPS |

### camera (カメラ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `z_min` | `3.0` | カメラ高さの最小値 (m) |
| `z_max` | `5.0` | カメラ高さの最大値 (m) |
| `hfov_deg` | `60.0` | 水平視野角 (度) |
| `image_size` | `[1280, 720]` | 画像サイズ |

### sampling (サンプリング設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `category_ratios.direct_net` | `0.05` | ネット直撃の割合 |
| `category_ratios.direct_fence` | `0.05` | フェンス直撃の割合 |
| `category_ratios.in_court` | `0.60` | インコートショットの割合 |
| `category_ratios.out_court` | `0.30` | アウトショットの割合 |
| `in_court_cell_weights` | `uniform` | インコートセルの重み付け |
| `out_court_cell_weights` | `uniform` | アウトコートセルの重み付け |
| `per_from_cell_samples` | `100` | セルあたりのサンプル数 |

### generator (ジェネレータ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `num_cameras_sampled` | `8` | サンプリングするカメラ数 |
| `ball_visibility_threshold` | `0.8` | ボール可視性閾値 |
| `max_attempts_per_cell` | `10000` | セルあたりの最大試行回数 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         generate_dataset.py                                  │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  ShotSimulator  │      │BLCSSceneGenerat │      │  BLCSDatasetWriter  │  │
│  │                 │──────▶│                 │──────▶│                     │  │
│  │ - 物理計算      │      │ - カメラ配置    │      │ - NPZ 保存          │  │
│  │ - 軌道生成      │      │ - 投影計算      │      │ - Split 情報        │  │
│  │ - イベント検出  │      │ - 可視性判定    │      │ - メタデータ        │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
│           ▲                                                                  │
│           │                                                                  │
│  ┌─────────────────┐                                                        │
│  │   CellManager   │                                                        │
│  │                 │                                                        │
│  │ - コート分割    │                                                        │
│  │ - ショットカテゴリ                                                        │
│  │ - サンプリング  │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. CellManager がコートをセルに分割し、ショットカテゴリを管理
2. 各セルからの発射点について:
   a. ShotSimulator がボール軌道を物理シミュレーション
   b. バウンス、ネット衝突などのイベントを検出
3. BLCSSceneGenerator が:
   a. ランダムなカメラ位置を生成
   b. 3D軌道を各カメラ視点に投影
   c. 可視性チェックを実行
4. BLCSDatasetWriter がシーンごとに NPZ ファイルを保存
```

## ショットカテゴリ

| カテゴリ | 説明 |
|---------|------|
| `DIRECT_NET` | ネットに直接当たるショット |
| `DIRECT_FENCE` | フェンスに直接当たるショット |
| `IN_COURT` | コート内にバウンドするショット |
| `OUT_COURT` | コート外にバウンドするショット |

## 出力構造

```
data/blcs/
├── config.yaml          # 使用した設定
├── meta.json            # メタデータ
├── dataset_info.json    # データセット統計
├── split_info.json      # Train/Val/Test 分割情報
└── scenes/
    ├── scene_000000.npz
    ├── scene_000001.npz
    └── ...
```

## NPZ ファイル内容

各シーンには以下のデータが含まれます:

- `ball_pos_world`: [T, 3] ワールド座標系でのボール位置
- `ball_pos_2d`: [C, T, 2] 各カメラでの2D投影位置
- `court_kp_2d`: [C, 20, 2] 各カメラでのコートキーポイント
- `camera_matrices`: [C, 3, 4] カメラ投影行列
- `visibility`: [C, T] 各カメラでの可視性フラグ
- `meta`: シーンメタデータ

## 関連モジュール

- `src.blcs.simulation.ball_physics`: 物理シミュレーション
- `src.blcs.simulation.shot_simulator`: ショットシミュレータ
- `src.blcs.simulation.cell_manager`: セル管理
- `src.blcs.generate_dataset.scene_generator`: シーン生成
- `src.blcs.generate_dataset.io.dataset_io`: データ入出力
