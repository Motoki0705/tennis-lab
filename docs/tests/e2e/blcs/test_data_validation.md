# test_data_validation.py (BLCS)

生成されたデータがBLCSの型定義に準拠することを検証するテストです。

**ファイル**: `tests/e2e/blcs/test_data_validation.py`

## 概要

このテストファイルは以下を検証します：

1. 生成NPZシーンファイルが `BLCSSceneMeta`, `BLCSCameraParams` スキーマに準拠
2. データセットサンプルが `BLCSSample` スキーマに準拠
3. バッチデータが `BLCSBatch` スキーマに準拠
4. テンソル形状、dtype、値範囲が正しい

---

## TestBLCSGeneratedSceneValidation

NPZシーンファイルの検証テストクラス。

### test_scene_meta_schema

**検証内容**: シーンメタデータが `BLCSSceneMeta` に準拠

**要件**:
- 必須キー: `scene_id`, `from_cell`, `from_side`, `category`, `to_cell`, `t_net`, `t_fence`, `t_bounce1`, `t_bounce2`, `fps_out`, `sim_fps`, `num_frames`, `num_cameras_sampled`, `num_cameras`
- `from_cell`: 0-11 の整数
- `from_side`: "near" or "far"
- `num_frames`, `fps_out`: > 0

### test_camera_params_schema

**検証内容**: カメラパラメータが `BLCSCameraParams` に準拠

**要件**:
- `center`: 3要素のリスト
- `R`: 3x3行列
- `f`, `cx`, `cy`: 数値
- `w`, `h`: 正の整数

### test_trajectory_data_shapes

**検証内容**: 3D軌道データの形状

**要件**:
| データ | 形状 |
|--------|------|
| `ball_pos_world` | (T, 3) |
| `ball_pos_norm` | (T, 3) |
| `ball_vel_world` | (T, 3) |

### test_camera_projection_data_shapes

**検証内容**: カメラ投影データの形状

**要件**:
| データ | 形状 |
|--------|------|
| `ball_uv` | (T, 2) |
| `ball_visible` | (T,) |
| `court_kp_uv` | (20, 2) |
| `court_kp_visible` | (20,) |

### test_visibility_ratio_in_range

**検証内容**: 可視性比率が [0, 1] 範囲内

---

## TestBLCSDatasetSampleValidation

`BallTrajectoryDataset` サンプルの検証テストクラス。

### test_sample_has_required_keys

**検証内容**: 必須キーの存在

**必須キー**: `ball_uv`, `ball_mask`, `court_kp`, `court_vis`, `position_3d`, `velocity_3d`, `seq_len`

### test_sample_schema_validation

**検証内容**: `BLCSSample` スキーマへの完全準拠

`validate_blcs_sample()` を使用した包括的検証。

### test_sample_uv_normalized

**検証内容**: UV座標が [0, 1] 範囲に正規化

**対象**: `ball_uv`, `court_kp`

### test_sample_tensor_dtypes

**検証内容**: テンソルが float32/float64

**対象**: `ball_uv`, `court_kp`, `position_3d`, `velocity_3d`

### test_seq_len_matches_tensor_lengths

**検証内容**: `seq_len` がテンソル長と一致

**要件**:
- `ball_uv.shape[0] == seq_len`
- `ball_mask.shape[0] == seq_len`

---

## TestBLCSDataLoaderBatchValidation

`DataLoader` バッチの検証テストクラス。

### test_batch_schema_validation

**検証内容**: `BLCSBatch` スキーマへの準拠

`validate_blcs_batch()` を使用。

### test_batch_shapes_consistent

**検証内容**: バッチ内の形状整合性

**要件**:
- すべての時間次元テンソルが同じ `T_max`
- `ball_mask`: (B, T_max)
- `position_3d`: (B, T_max, 3)
- `velocity_3d`: (B, T_max, 3)
- `court_kp`: (B, 20, 2)

### test_batch_padding_valid

**検証内容**: パディングが正しく適用

**要件**: `seq_len[i] <= T_max` for all samples
