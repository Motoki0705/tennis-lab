# PLCS visualize

PLCSシーンの可視化および学習済みモデルによる予測結果の確認を行うスクリプト。

## 概要

このスクリプトは、生成されたPLCSシーンデータの可視化や、学習済みモデルを使用した予測結果の可視化を行います。3Dビュー、2Dトップダウンビュー、カメラビュー、アニメーションなど、複数の表示モードをサポートします。

## コマンド例

```bash
# デフォルト設定で可視化
uv run python -m src.tasks.plcs.scripts.visualize

# シーンパスと情報表示を指定
uv run python -m src.tasks.plcs.scripts.visualize visualization.scene_path=data/plcs/scenes/scene_000000.npz visualization.info=true

# 特定のフレームとカメラを指定
uv run python -m src.tasks.plcs.scripts.visualize visualization.frame=10 visualization.camera=2

# アニメーションを保存
uv run python -m src.tasks.plcs.scripts.visualize visualization.view=animation visualization.save=output.mp4

# 予測モードで実行
uv run python -m src.tasks.plcs.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/plcs/frame/logs/version_0/checkpoints/last.ckpt

# 3Dビューを表示
uv run python -m src.tasks.plcs.scripts.visualize visualization.view=3d
```

## コンフィグ

エントリポイント: `src/tasks/plcs/configs/visualize.yaml`

### defaults 構成

```yaml
defaults:
  - visualization: default
```

### visualization (可視化設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mode` | `visualize` | モード (visualize/predict) |
| `scene_path` | `data/plcs/scenes/scene_000001.npz` | シーンファイルパス |
| `frame` | `0` | 表示するフレームインデックス |
| `view` | `multi` | 表示モード |
| `camera` | `0` | カメラインデックス |
| `animation_view` | `2d_topdown` | アニメーションビュータイプ |
| `fps` | `null` | アニメーションFPS (nullでシーンのFPSを使用) |
| `save` | `null` | 出力ファイルパス |
| `save_input` | `null` | 入力シーンアニメーション保存パス |
| `info` | `false` | シーン情報を表示するか |
| `checkpoint` | `outputs/plcs/frame/logs/version_0/checkpoints/last.ckpt` | 予測モード用チェックポイント |
| `device` | `auto` | デバイス (auto/cuda/cpu) |

### view オプション

| 値 | 説明 |
|----|------|
| `multi` | 複数ビューを同時表示 (3D + 2D + Camera) |
| `3d` | 3Dパースペクティブビュー |
| `2d` | 2Dトップダウンビュー |
| `camera` | カメラ視点ビュー |
| `animation` | アニメーション表示/保存 |

### animation_view オプション

| 値 | 説明 |
|----|------|
| `2d_topdown` | 2Dトップダウンアニメーション |
| `2d_camera` | 2Dカメラビューアニメーション |
| `3d` | 3Dアニメーション |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             visualize.py                                     │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  mode=visualize │                                                        │
│  │                 │                                                        │
│  │  load_scene()   │──▶ SceneRenderer ──▶ matplotlib 表示/保存              │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐                               │
│  │  mode=predict   │      │                 │                               │
│  │                 │      │  PLCSモデル     │                               │
│  │  load_scene()   │──▶   │  (checkpoint)   │──▶ 予測結果可視化             │
│  └─────────────────┘      └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー (visualize モード):
1. NPZ シーンファイルを読み込み
2. (info=true の場合) シーンメタデータを表示
3. 指定されたビューでレンダリング
4. (save が指定された場合) ファイルに保存、なければ表示

処理フロー (predict モード):
1. NPZ シーンファイルを読み込み
2. チェックポイントからモデルをロード
3. 入力キーポイントで推論
4. Ground Truth と予測結果を比較可視化
```

## 表示内容

### シーン情報 (info=true)

```
============================================================
Scene Information
============================================================
Scene ID:        scene_000001
Motion source:   data/ACCAD/Female1Running_c3d/...
Category:        running
Gender:          female
FPS:             30
Num frames:      150
Duration:        5.00 seconds
Initial pos:     (0.50, 0.30)
Initial yaw:     45.0°
Cameras sampled: 10
Cameras kept:    5

Position statistics (normalized):
  X range: [0.200, 0.800]
  Y range: [0.150, 0.650]
  Z range: [0.000, 0.020]

Camera visibility:
  Camera 0: Human 95.0%, Court 18.0/20
  Camera 1: Human 92.0%, Court 17.0/20
  ...
```

## 出力例

- 静止画: PNG ファイル
- アニメーション: MP4 ファイル

## 関連モジュール

- `src.tasks.plcs.generate_dataset.io.dataset_io`: シーン読み込み
- `src.utils.rendering.PLCSSceneRenderer`: シーンレンダリング
