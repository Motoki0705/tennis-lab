# BLCS generate_dataset

画面上のシーケンシャルなボールの位置とコートのキーポイントを用いて3D上のボールの軌道を推定するための合成データセットを生成するスクリプト。

## 概要

このスクリプトは、テニスボールの物理シミュレーション（重力、空気抵抗、マグナス効果、バウンス）を使用して、様々なショットパターンのボール軌道データを生成します。生成されたデータは、2Dボール位置シーケンスから3D軌道を推定するモデルの学習に使用されます。

### 生成モード

- **Shot モード** (デフォルト): 単発ショットを生成。コート上の各セルから発射し、2回目のバウンドまたはフェンス到達で終了。
- **Rally モード**: 複数ショットを連結したラリーを生成。1バウンド後〜2バウンド前のタイミングで打ち返しを行い、ネット/アウト/最大ラリー数到達で終了。

## コマンド例

```bash
# デフォルト設定で実行 (Shot モード)
uv run python -m src.blcs.scripts.generate_dataset

# 出力先とサンプル数を指定
uv run python -m src.blcs.scripts.generate_dataset run.output_dir=data/blcs sampling.per_from_cell_samples=10

# Rally モードで実行
uv run python -m src.blcs.scripts.generate_dataset generator.mode=rally generator.num_rally_scenes=100

# Rally モードでラリー設定をカスタマイズ
uv run python -m src.blcs.scripts.generate_dataset generator.mode=rally rally.max_rallies=5 rally.court_margin=1.0

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
  - rally: default
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

### rally (ラリー設定) - Rally モード専用

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_rallies` | `10` | 最大ラリー数（ショット数） |
| `max_total_frames` | `12000` | ラリー全体の最大フレーム数 |
| `court_margin` | `0.5` | コート外判定の余裕 (m) |
| `hit_timing_range` | `[0.2, 0.8]` | 打ち返しタイミング範囲（1st〜2ndバウンド間の割合） |
| `return_z_range` | `[0.8, 1.4]` | 打ち返し時の高さ範囲 (m) |

### camera (カメラ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `z_min` | `3.0` | カメラ高さの最小値 (m) |
| `z_max` | `5.0` | カメラ高さの最大値 (m) |
| `hfov_deg` | `60.0` | 水平視野角 (度) |
| `image_size` | `[1280, 720]` | 画像サイズ |

### sampling (サンプリング設定) - Shot モード専用

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
| `mode` | `shot` | 生成モード (`shot` または `rally`) |
| `num_rally_scenes` | `1000` | Rally モードでの生成シーン数 |
| `num_cameras_sampled` | `8` | サンプリングするカメラ数 |
| `ball_visibility_threshold` | `0.8` | ボール可視性閾値 |
| `max_attempts_per_cell` | `10000` | セルあたりの最大試行回数 |

## アーキテクチャ・フロー

### Shot モード

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
└─────────────────────────────────────────────────────────────────────────────┘
```

### Rally モード

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         generate_dataset.py                                  │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │ RallySimulator  │      │BLCSSceneGenerat │      │  BLCSDatasetWriter  │  │
│  │                 │──────▶│                 │──────▶│                     │  │
│  │ - ショット連結  │      │ - カメラ配置    │      │ - NPZ 保存          │  │
│  │ - 打ち返し生成  │      │ - 投影計算      │      │ - Split 情報        │  │
│  │ - ラリー終了判定│      │ - 可視性判定    │      │ - メタデータ        │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
│           ▲                                                                  │
│           │                                                                  │
│  ┌─────────────────┐                                                        │
│  │  ShotSimulator  │  (初期条件生成に使用)                                   │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Rally 処理フロー:
1. 開始位置から初回ショットを生成
2. 1st バウンド後、2nd バウンド前のランダムなタイミングで打ち返し
3. 相手コートに向けて新たな初期条件でショット生成
4. ラリー終了条件まで繰り返し:
   - NET_FAULT: ネット直撃
   - OUT: コート外+margin でバウンド
   - MAX_RALLIES: 最大ラリー数到達
   - MAX_FRAMES: 最大フレーム数到達
   - DOUBLE_BOUNCE: 2バウンド（正常終了）
```

## ショットカテゴリ (Shot モード)

| カテゴリ | 説明 |
|---------|------|
| `DIRECT_NET` | ネットに直接当たるショット |
| `DIRECT_FENCE` | フェンスに直接当たるショット |
| `IN_COURT` | コート内にバウンドするショット |
| `OUT_COURT` | コート外にバウンドするショット |

## ラリー終了理由 (Rally モード)

| 終了理由 | 説明 |
|---------|------|
| `net_fault` | ボールがネットに当たった |
| `out` | ボールがコート外（+margin）にバウンドした |
| `double_bounce` | 2バウンド（正常な得点終了） |
| `max_rallies` | 最大ラリー数に到達した |
| `max_frames` | 最大フレーム数に到達した |

## 出力構造

```
data/blcs/
├── config.yaml          # 使用した設定
├── meta.json            # メタデータ
├── dataset_info.json    # データセット統計
├── split_info.json      # Train/Val/Test 分割情報
└── scenes/
    ├── scene_000000.npz  # Shot モード
    ├── rally_000000.npz  # Rally モード
    └── ...
```

## NPZ ファイル内容

### Shot モード

各シーンには以下のデータが含まれます:

- `meta`: シーンメタデータ (JSON)
- `ball_pos_world`: [T, 3] ワールド座標系でのボール位置
- `ball_pos_norm`: [T, 3] 正規化されたボール位置
- `ball_vel_world`: [T, 3] ボール速度
- `num_cameras`: カメラ数
- `cam_{i}_params`: カメラパラメータ
- `cam_{i}_ball_uv`: [T, 2] 2D投影位置
- `cam_{i}_ball_visible`: [T] 可視性フラグ
- `cam_{i}_court_kp_uv`: [20, 2] コートキーポイント

### Rally モード

Shot モードに加えて以下のデータが含まれます:

- `rally_length`: ラリー内のショット数
- `end_reason`: ラリー終了理由
- `meta.shots`: 各ショットのイベント情報
  - `shot_index`: ショット番号 (0-indexed)
  - `from_side`: 打ち手側 ("near" or "far")
  - `t_start`, `t_net`, `t_bounce1`, `t_bounce2`, `t_return`: フレームタイミング

## 関連モジュール

- `src.blcs.simulation.ball_physics`: 物理シミュレーション
- `src.blcs.simulation.shot_simulator`: ショットシミュレータ
- `src.blcs.simulation.rally_simulator`: ラリーシミュレータ
- `src.blcs.simulation.cell_manager`: セル管理
- `src.blcs.generate_dataset.scene_generator`: シーン生成
- `src.blcs.generate_dataset.io.dataset_io`: データ入出力
