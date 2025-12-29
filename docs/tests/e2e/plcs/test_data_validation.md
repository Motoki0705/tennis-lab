# test_data_validation.py (PLCS)

生成されたデータがPLCSの型定義に準拠することを検証するテストです。

**ファイル**: `tests/e2e/plcs/test_data_validation.py`

## 概要

このテストファイルは以下を検証します：

1. 生成NPZシーンファイルが `PLCSSceneMeta`, `PLCSCameraParams` スキーマに準拠
2. データセットサンプルが `PLCSFrameBatch` スキーマに準拠
3. テンソル形状、dtype、値範囲が正しい

---

## TestPLCSGeneratedSceneValidation

NPZシーンファイルの検証テストクラス。

### test_scene_meta_schema

**検証内容**: シーンメタデータが `PLCSSceneMeta` に準拠

**要件**:
- `scene_id`, `motion_source`, `motion_category`: 文字列
- `gender`: "male", "female", "neutral" のいずれか
- `fps`, `num_frames`: 整数
- `initial_position`: リスト
- `initial_yaw`: 数値

### test_camera_params_schema

**検証内容**: カメラパラメータが `PLCSCameraParams` に準拠

BLCSと同じスキーマを使用。

### test_position_data_shapes

**検証内容**: 位置・回転データの形状

**要件**:
| データ | 形状 |
|--------|------|
| `position` | (T, 3) |
| `rotation` | (T, 2) |

### test_rotation_normalized

**検証内容**: 回転値が [-1, 1] 範囲（sin/cos表現）

### test_camera_keypoint_data_shapes

**検証内容**: カメラキーポイントデータの形状

**要件**:
| データ | 形状 |
|--------|------|
| `human_kp_uv` | (T, 17, 2) |
| `human_kp_visible` | (T, 17) |
| `court_kp_uv` | (T, 20, 2) |
| `court_kp_visible` | (T, 20) |

### test_visibility_ratio_in_range

**検証内容**: 可視性比率が [0, 1] 範囲内

---

## TestPLCSDatasetSampleValidation

`SceneDataset` サンプルの検証テストクラス。

### test_sample_has_required_keys

**検証内容**: 必須キーの存在

**必須キー**: `human_kp`, `court_kp`, `human_vis`, `court_vis`, `position`, `rotation`

### test_sample_schema_validation

**検証内容**: `PLCSFrameBatch` スキーマへの準拠

`validate_plcs_frame_batch()` を使用。

### test_sample_tensor_shapes

**検証内容**: テンソル形状

**要件**:
| キー | 形状 | 説明 |
|-----|------|------|
| `human_kp` | (34,) | 17×2 フラット化 |
| `court_kp` | (40,) | 20×2 フラット化 |
| `position` | (3,) | - |
| `rotation` | (2,) | - |

### test_sample_rotation_normalized

**検証内容**: 回転が [-1, 1] 範囲（sin/cos）

### test_sample_visibility_masks_valid

**検証内容**: 可視性マスクが 0/1 のみ

**対象**: `human_vis`, `court_vis`

### test_sample_tensor_dtypes

**検証内容**: テンソルが float32/float64

**対象**: `human_kp`, `court_kp`, `position`, `rotation`

---

## TestPLCSDataLoaderBatchValidation

`DataLoader` バッチの検証テストクラス。

### test_batch_shapes_consistent

**検証内容**: バッチ次元の整合性

すべてのテンソルが同じバッチサイズ `B` を持つ。

### test_batch_tensor_shapes

**検証内容**: バッチ化されたテンソルの形状

**要件**:
| キー | 形状 |
|-----|------|
| `human_kp` | (B, 34) |
| `court_kp` | (B, 40) |
| `position` | (B, 3) |
| `rotation` | (B, 2) |
