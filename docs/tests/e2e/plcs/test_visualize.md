# test_visualize.py (PLCS)

可視化スクリプトのテストです。

**ファイル**: `tests/e2e/plcs/test_visualize.py`

**対象スクリプト**: `src.plcs.scripts.visualize`

## 概要

このテストファイルは以下を検証します：

1. 可視化スクリプトが正常終了すること
2. 出力画像が生成されること
3. 各モード（フレーム/シーケンス予測）が動作すること

---

## test_plcs_visualize_ground_truth

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

---

## test_plcs_visualize_predict

フレームモデル予測可視化テスト。

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

- `create_minimal_plcs_checkpoint` で生成したチェックポイント

---

## test_plcs_visualize_sequence_predict

シーケンスモデル予測可視化テスト。

### 検証内容

1. `mode=predict-seq` で正常終了
2. 出力PNG画像が生成される

### 使用される設定

```bash
visualization.scene_path={scene_path}
visualization.mode=predict-seq
visualization.checkpoint={checkpoint_path}
visualization.save={output_path}
```

### 前提条件

- `create_minimal_plcs_sequence_checkpoint` で生成したチェックポイント

---

## test_plcs_visualize_different_views

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

## 可視化モード

| モード | 説明 | 必要なもの |
|--------|------|-----------|
| `visualize` | グラウンドトゥルースの表示 | シーンファイル |
| `predict` | フレームモデル予測の表示 | シーン + フレームモデルチェックポイント |
| `predict-seq` | シーケンスモデル予測の表示 | シーン + シーケンスモデルチェックポイント |

## ビューオプション

| ビュー | 説明 |
|--------|------|
| `3d` | 3D空間でのプレーヤー位置表示 |
| `2d` | コート平面（上から見た図） |

## タイムアウト

120秒
