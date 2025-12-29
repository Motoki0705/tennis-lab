# test_data_validation.py (WASB)

WASBデータセットの構造とサンプルの検証テストです。

**ファイル**: `tests/e2e/wasb/test_data_validation.py`

## 概要

このテストファイルは以下を検証します：

1. データセットディレクトリ構造が正しいこと
2. Label.csv のフォーマットが正しいこと
3. `BallDetectionSample` スキーマに準拠すること
4. テンソル形状、dtype、値範囲が正しいこと

---

## TestWASBDatasetValidation

データセットディレクトリ構造の検証テストクラス。

### test_dataset_structure

**検証内容**: ゲーム/クリップディレクトリ構造

**要件**:
- ゲームディレクトリが存在
- 各ゲームにクリップディレクトリが存在
- 各クリップに `Label.csv` が存在
- 各クリップに画像ファイル（.jpg/.png）が存在

### test_label_csv_format

**検証内容**: Label.csv のフォーマット

**要件**:
- ヘッダーに `file name`, `visibility`, `x-coordinate`, `y-coordinate` を含む
- 各行に4つ以上のフィールドがある

---

## TestWASBBallDetectionSampleValidation

`BallDetectionSequenceDataset` サンプルの検証テストクラス。

### test_sample_has_required_keys

**検証内容**: `BallDetectionSample` の必須キーが存在

**必須キー**:
- `frames`: 入力フレーム
- `targets_px`: ピクセル座標ターゲット
- `targets_norm`: 正規化座標ターゲット
- `target_heatmaps`: ターゲットヒートマップ
- `visibility`: 可視性フラグ
- `scores`: 信頼度スコア
- `match`: マッチID
- `clip`: クリップID

### test_sample_tensor_shapes

**検証内容**: テンソル形状

**要件**:
| キー | 形状 | 説明 |
|-----|------|------|
| `frames` | (T, C, H, W) | T=frames_in, C=3 |
| `targets_px` | (frames_out, 2) | - |
| `targets_norm` | (frames_out, 2) | - |
| `visibility` | (frames_out,) | - |

### test_sample_normalized_targets_in_range

**検証内容**: 正規化ターゲットが [0, 1] 範囲

可視ターゲットのみを検証（`visibility > 0`）。

### test_sample_heatmap_shape_matches_frame

**検証内容**: ヒートマップサイズがフレームサイズと整合

ヒートマップの空間次元 ≤ フレームの空間次元。

### test_sample_visibility_values

**検証内容**: 可視性値が有効

**許可される値**: 0, 1, 2
- 0: 不可視
- 1: 可視
- 2: オクルージョン

### test_sample_tensor_dtypes

**検証内容**: テンソルdtype

**要件**:
- `frames`: float32/float64
- `targets_px`: float32/float64
- `target_heatmaps`: float32/float64

### test_sample_metadata_types

**検証内容**: メタデータの型

**要件**:
- `match`: str
- `clip`: str

---

## TestWASBDataLoaderBatchValidation

バッチデータの検証テストクラス。

### test_batch_shapes_consistent

**検証内容**: バッチ次元が一貫している

すべてのテンソルが同じバッチサイズ `B` を持つ。

### test_batch_frames_shape

**検証内容**: バッチ化されたフレームの形状

**要件**: `frames` が5D (B, T, C, H, W)
