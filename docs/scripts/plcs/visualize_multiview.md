# PLCS visualize_multiview.py

マルチビュー（複数カメラ）からの観測を使用してプレイヤーの3D位置・回転を推定し、可視化するスクリプトです。

## 概要

`visualize_multiview.py` は、複数カメラからの観測を統合してプレイヤーの位置・姿勢を推定し、
その結果を様々な視点で可視化します。

**シーケンシャル推論対応**: マルチビューモデルは常にシーケンシャル入力 `(N_cam, T, ...)` を
期待し、時系列全体の位置・回転を出力します。

## 実行方法

### 基本実行（Ground Truth可視化）

```bash
uv run python -m src.plcs.scripts.visualize_multiview
```

### シーン情報の表示

```bash
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.scene_path=data/plcs/scenes/scene_000001.npz \
    visualization.info=true
```

### マルチビュー予測と可視化

```bash
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.scene_path=data/plcs/scenes/scene_000001.npz \
    visualization.mode=predict \
    visualization.checkpoint=outputs/plcs/multiview/logs/version_0/checkpoints/last.ckpt \
    visualization.save=output.png
```

### 使用するカメラの指定

```bash
# 特定のカメラのみ使用
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.cameras="0,1,2"

# すべてのカメラを使用（デフォルト）
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.cameras=all
```

### アニメーション出力

```bash
uv run python -m src.plcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.view=animation \
    visualization.animation_view=3d \
    visualization.save=output.mp4
```

## 設定オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `visualization.mode` | `visualize` | モード（visualize/predict） |
| `visualization.scene_path` | - | シーンファイルパス |
| `visualization.frame` | `0` | 静止画表示時のフレーム番号 |
| `visualization.view` | `multi` | 表示モード（multi/3d/2d/animation） |
| `visualization.cameras` | `all` | 使用するカメラ（all/カンマ区切り） |
| `visualization.animation_view` | `3d` | アニメーション視点（3d/2d） |
| `visualization.fps` | `null` | アニメーションFPS（null=シーンのFPS） |
| `visualization.save` | `null` | 出力ファイルパス |
| `visualization.info` | `false` | シーン情報を表示して終了 |
| `visualization.checkpoint` | - | チェックポイントパス |
| `visualization.device` | `auto` | 推論デバイス（auto/cuda/cpu） |

## 関連ファイル

- Predictor: `src/plcs/inference/multiview_predictor.py`
- Config: `src/plcs/configs/visualize_multiview.yaml`
- 設定詳細: `src/plcs/configs/visualization/multiview.yaml`
