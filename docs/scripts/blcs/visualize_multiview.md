# BLCS visualize_multiview.py

マルチビュー（複数カメラ）からのボール観測を使用して3D軌道を推定し、可視化するスクリプトです。

## 概要

`visualize_multiview.py` は、複数カメラからの2Dボール観測を統合して3D軌道を推定し、
その結果を様々な視点で可視化します。

## 実行方法

### 基本実行（Ground Truth可視化）

```bash
uv run python -m src.blcs.scripts.visualize_multiview
```

### シーン情報の表示

```bash
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.scene_path=data/blcs/scenes/scene_000003.npz \
    visualization.info=true
```

### マルチビュー予測と可視化

```bash
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.scene_path=data/blcs/scenes/scene_000003.npz \
    visualization.mode=predict \
    visualization.checkpoint=outputs/blcs_multiview/checkpoints/last.ckpt \
    visualization.save=output.png
```

### 使用するカメラの指定

```bash
# 特定のカメラのみ使用
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.cameras="0,1,2"

# すべてのカメラを使用（デフォルト）
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.cameras=all
```

### 予測結果のエクスポート

```bash
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.output=predictions.json  # または predictions.pt
```

### 比較アニメーション出力

```bash
uv run python -m src.blcs.scripts.visualize_multiview \
    visualization.mode=predict \
    visualization.view=animation \
    visualization.animation_view=3d \
    visualization.save=comparison.mp4
```

## 設定オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `visualization.mode` | `visualize` | モード（visualize/predict） |
| `visualization.scene_path` | - | シーンファイルパス |
| `visualization.frame` | `0` | 静止画表示時のフレーム番号 |
| `visualization.view` | `multi` | 表示モード（multi/3d/2d/camera/animation） |
| `visualization.cameras` | `all` | 使用するカメラ（all/カンマ区切り） |
| `visualization.animation_view` | `3d` | アニメーション視点（3d/2d/camera） |
| `visualization.fps` | `null` | アニメーションFPS（null=シーンのFPS） |
| `visualization.save` | `null` | 出力ファイルパス |
| `visualization.info` | `false` | シーン情報を表示して終了 |
| `visualization.checkpoint` | - | チェックポイントパス |
| `visualization.output` | `null` | 予測結果エクスポートパス（.pt/.json） |

## 出力形式

### JSON形式

```json
{
  "position": [[x, y, z], ...],
  "position_meters": [[x, y, z], ...]
}
```

### PyTorch形式（.pt）

```python
{
  "position": torch.Tensor,  # (T, 3)
  "position_meters": torch.Tensor  # (T, 3)
}
```

## 関連ファイル

- Predictor: `src/blcs/inference/multiview_predictor.py`
- Config: `src/blcs/configs/visualize_multiview.yaml`
- 設定詳細: `src/blcs/configs/visualization/multiview.yaml`
