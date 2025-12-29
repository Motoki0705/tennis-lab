# test_visualize.py (BLCS)

可視化スクリプトのテストです。

**ファイル**: `tests/e2e/blcs/test_visualize.py`

**対象スクリプト**: `src.blcs.scripts.visualize`

## 概要

このテストファイルは以下を検証します：

1. 可視化スクリプトが正常終了すること
2. 出力画像が生成されること
3. 各ビューモードが動作すること

---

## test_blcs_visualize_ground_truth

グラウンドトゥルース可視化テスト。

### 検証内容

1. `mode=visualize` で正常終了
2. 出力PNG画像が生成される

### 使用される設定

```bash
visualization.scene_path={scene_path}
visualization.mode=visualize
visualization.view=3d
visualization.save={output_path}
```

### 前提条件

- `create_minimal_blcs_dataset` で生成したシーン

---

## test_blcs_visualize_predict

モデル予測可視化テスト。

### 検証内容

1. `mode=predict` で正常終了
2. 出力PNG画像が生成される

### 使用される設定

```bash
visualization.scene_path={scene_path}
visualization.mode=predict
visualization.checkpoint={checkpoint_path}
visualization.save={output_path}
visualization.view=3d
```

### 前提条件

- `create_minimal_blcs_dataset` で生成したシーン
- `create_minimal_blcs_checkpoint` で生成したチェックポイント

---

## test_blcs_visualize_different_views

異なるビューモードのテスト。

### 検証内容

1. `view=2d` で正常終了
2. 出力PNG画像が生成される

### 使用される設定

```bash
visualization.scene_path={scene_path}
visualization.mode=visualize
visualization.view=2d
visualization.save={output_path_2d}
```

---

## test_blcs_visualize_camera_view

カメラビューモードのテスト。

### 検証内容

1. `view=camera` で正常終了
2. 出力PNG画像が生成される

### 使用される設定

```bash
visualization.scene_path={scene_path}
visualization.mode=visualize
visualization.view=camera
visualization.camera=0
visualization.save={output_path_camera}
```

---

## 可視化モード

| モード | 説明 | 必要なもの |
|--------|------|-----------|
| `visualize` | グラウンドトゥルースの表示 | シーンファイル |
| `predict` | モデル予測の表示 | シーンファイル + チェックポイント |

## ビューオプション

| ビュー | 説明 |
|--------|------|
| `3d` | 3D空間での軌道表示 |
| `2d` | 2D平面（上から見た図） |
| `camera` | カメラ視点（`camera=N` で指定） |

## タイムアウト

120秒
