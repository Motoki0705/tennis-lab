# PLCS generate_dataset

画面上のプレーヤーのキーポイントとコートのキーポイントを用いて3D上のプレーヤーの位置と回転を推定するための合成データセットを生成するスクリプト。

## 概要

このスクリプトは、SMPL-H モーションキャプチャデータを使用して、仮想カメラから見た人物のキーポイントとコートのキーポイントのシミュレーションデータを生成します。生成されたデータは、カメラ視点から3D位置・回転を推定するモデルの学習に使用されます。

## コマンド例

```bash
# デフォルト設定で実行
uv run python -m src.plcs.scripts.generate_dataset

# 出力先とシーン数を指定
uv run python -m src.plcs.scripts.generate_dataset run.output_dir=data/plcs simulation.num_scenes=10

# カテゴリを指定して生成
uv run python -m src.plcs.scripts.generate_dataset run.category=running

# シード値を変更
uv run python -m src.plcs.scripts.generate_dataset run.seed=123
```

## コンフィグ

エントリポイント: `src/plcs/configs/generate_dataset.yaml`

### defaults 構成

```yaml
defaults:
  - paths: default
  - simulation: default
  - camera: default
  - motion_sources: default
  - run: generate_dataset
```

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `output_dir` | `data/plcs` | 出力ディレクトリ |
| `seed` | `42` | 乱数シード |
| `device` | `auto` | デバイス (auto/cuda/cpu) |
| `category` | `null` | 生成するモーションカテゴリ (null=全カテゴリ) |

### simulation (シミュレーション設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `num_scenes` | `3000` | 生成するシーン数 |
| `num_cameras` | `5` | シーンあたりのカメラ数 |
| `human_visibility_threshold` | `0.8` | 人物の可視性閾値 |
| `court_visibility_threshold` | `15` | コートのキーポイント可視性閾値 |

### camera (カメラ設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `z_min` | `3.0` | カメラ高さの最小値 (m) |
| `z_max` | `5.0` | カメラ高さの最大値 (m) |
| `hfov_deg` | `60.0` | 水平視野角 (度) |
| `image_size` | `[1280, 720]` | 画像サイズ |

### motion_sources (モーションソース)

```yaml
running:
  paths:
    - data/ACCAD/Female1Running_c3d
    - data/ACCAD/Male1Running_c3d
    # ...
  weight: 0.5

walking:
  paths:
    - data/ACCAD/Female1Walking_c3d
    # ...
  weight: 0.4

general:
  paths:
    - data/ACCAD/Female1General_c3d
    # ...
  weight: 0.1
```

### paths (パス設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `smplh_model_path` | `data/smplx/smplh` | SMPL-H モデルパス |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         generate_dataset.py                                  │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │   MotionSampler │──────▶│  SceneGenerator │──────▶│  PLCSDatasetWriter │  │
│  │                 │      │                 │      │                     │  │
│  │ - SMPL-H models │      │ - カメラ配置    │      │ - NPZ 保存          │  │
│  │ - モーション読込│      │ - 投影計算      │      │ - メタデータ保存    │  │
│  │ - カテゴリ抽出  │      │ - 可視性判定    │      │ - 統計情報保存      │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. MotionSampler が ACCAD などのモーションデータから SMPL-H パラメータをロード
2. SceneGenerator が:
   - ランダムなカメラ位置を生成
   - 人物の 3D キーポイントをカメラ座標系に投影
   - コートのキーポイントを投影
   - 可視性チェックを実行
3. PLCSDatasetWriter がシーンごとに NPZ ファイルを保存
```

## 出力構造

```
data/plcs/
├── config.yaml          # 使用した設定
├── stats.json           # 生成統計
├── scenes_meta.json     # 全シーンのメタデータ
├── dataset_info.json    # データセット情報
└── scenes/
    ├── scene_000000.npz
    ├── scene_000001.npz
    └── ...
```

## 関連モジュール

- `src.plcs.generate_dataset.sampling.motion_sampler`: モーションデータのサンプリング
- `src.plcs.generate_dataset.scene_generator`: シーン生成ロジック
- `src.plcs.generate_dataset.io.dataset_io`: データ入出力
