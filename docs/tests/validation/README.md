# Validation ユーティリティ

E2Eテストで使用するスキーマ検証およびテンソル検証ユーティリティの詳細ドキュメントです。

## 概要

`tests/e2e/validation/` には2つの検証モジュールがあります：

| モジュール | 説明 |
|-----------|------|
| `tensor_validators.py` | テンソルの形状・dtype・値範囲を検証する低レベルユーティリティ |
| `schema_validators.py` | TypedDict/dataclassスキーマに対する高レベル検証 |

## テンソルバリデータ (`tensor_validators.py`)

### 基本関数

#### `validate_tensor_shape(tensor, expected_shape, name)`

テンソル形状を検証します。動的次元には `None` を使用。

```python
from tests.e2e.validation import validate_tensor_shape

# 固定形状の検証
err = validate_tensor_shape(tensor, (32, 3), "positions")

# 動的次元を含む形状の検証（T は任意の長さ）
err = validate_tensor_shape(tensor, (None, 2), "ball_uv")  # (T, 2)

# バッチ+時間次元が動的
err = validate_tensor_shape(tensor, (None, None, 3), "batch_positions")  # (B, T, 3)
```

**戻り値**: エラーがあれば文字列、なければ `None`

#### `validate_tensor_dtype(tensor, expected_dtype, name)`

テンソルのdtypeを検証します。

```python
from tests.e2e.validation import validate_tensor_dtype
import torch

# 単一dtype
err = validate_tensor_dtype(tensor, torch.float32, "positions")

# 複数dtype許可
err = validate_tensor_dtype(tensor, [torch.float32, torch.float64], "positions")
```

#### `validate_tensor_range(tensor, min_val, max_val, name, *, allow_nan=False)`

テンソル値が指定範囲内かを検証します。

```python
from tests.e2e.validation import validate_tensor_range

# [0, 1] 範囲の検証
err = validate_tensor_range(tensor, 0.0, 1.0, "normalized_coords")

# 下限のみ検証
err = validate_tensor_range(tensor, 0.0, None, "positive_values")

# NaN許可
err = validate_tensor_range(tensor, 0.0, 1.0, "with_missing", allow_nan=True)
```

**チェック項目**:
- NaN の検出（`allow_nan=False` の場合）
- Inf の検出（常にエラー）
- 最小値・最大値の範囲

#### `validate_normalized_uv(tensor, name, *, strict=False)`

正規化UV座標（[0, 1]範囲）を検証します。

```python
from tests.e2e.validation import validate_normalized_uv

# 数値誤差を許容（±0.01）
err = validate_normalized_uv(tensor, "ball_uv")

# 厳密に [0, 1] を要求
err = validate_normalized_uv(tensor, "ball_uv", strict=True)
```

**要件**:
- 最後の次元が2（UV座標）
- 値が [0, 1] または [0-margin, 1+margin]（`strict=False`の場合 margin=0.01）

#### `validate_visibility_mask(tensor, name)`

可視性マスクを検証します。

```python
from tests.e2e.validation import validate_visibility_mask

err = validate_visibility_mask(tensor, "ball_mask")
```

**許可される値**:
- `torch.bool` 型
- 数値型で 0/1 のみ

#### `validate_dict_has_keys(data, required_keys, name)`

辞書に必須キーがあるかを検証します。

```python
from tests.e2e.validation import validate_dict_has_keys

errors = validate_dict_has_keys(
    sample,
    ["ball_uv", "ball_mask", "court_kp", "position_3d"],
    "BLCSSample"
)
```

**戻り値**: 欠落キーのエラーメッセージリスト

#### `collect_errors(*errors)`

複数の検証結果から非Noneエラーを収集します。

```python
from tests.e2e.validation import collect_errors

errors = collect_errors(
    validate_tensor_shape(t1, (32, 3), "t1"),
    validate_tensor_shape(t2, (32, 2), "t2"),
    validate_tensor_dtype(t3, torch.float32, "t3"),
)
# errors = ["t1: dim 1 expected 3, got 2"] など
```

---

## スキーマバリデータ (`schema_validators.py`)

### BLCS スキーマ

#### `validate_blcs_sample(sample) -> list[str]`

`BLCSSample` TypedDict への準拠を検証します。

**期待されるスキーマ** (from `src/blcs/data/types.py`):

| キー | 形状 | 説明 |
|-----|------|------|
| `ball_uv` | (T, 2) | 正規化UVでのボール2D軌道 |
| `ball_mask` | (T,) | ボール可視性マスク |
| `court_kp` | (20, 2) | コート2Dキーポイント（正規化UV） |
| `court_vis` | (20,) | コートキーポイント可視性 |
| `position_3d` | (T, 3) | グラウンドトゥルース3D軌道（正規化） |
| `velocity_3d` | (T, 3) | 3D速度ベクトル |
| `seq_len` | scalar | 実際のシーケンス長 |

**検証項目**:
- 必須キーの存在
- テンソル形状（`seq_len` に基づく）
- dtype（float32/float64）
- UV座標の正規化（[0, 1]）
- 可視性マスクの値（0/1）

#### `validate_blcs_batch(batch, batch_size=None) -> list[str]`

`BLCSBatch` TypedDict への準拠を検証します。

**期待されるスキーマ**:

| キー | 形状 | 説明 |
|-----|------|------|
| `ball_uv` | (B, T_max, 2) | パディング済みボール軌道 |
| `ball_mask` | (B, T_max) | パディング済み可視性マスク |
| `court_kp` | (B, 20, 2) | コートキーポイント |
| `court_vis` | (B, 20) | コートキーポイント可視性 |
| `position_3d` | (B, T_max, 3) | パディング済みグラウンドトゥルース |
| `velocity_3d` | (B, T_max, 3) | パディング済み速度 |
| `seq_len` | (B,) | 各サンプルの実際のシーケンス長 |

#### `validate_blcs_scene_meta(meta) -> list[str]`

`BLCSSceneMeta` への準拠を検証します。

**必須キー**:
- `scene_id` (str)
- `from_cell` (int, 0-11)
- `from_side` ("near" or "far")
- `category` (str)
- `to_cell` (int)
- `t_net`, `t_fence`, `t_bounce1`, `t_bounce2` (int)
- `fps_out`, `sim_fps`, `num_frames` (int, > 0)
- `num_cameras_sampled`, `num_cameras` (int)

#### `validate_blcs_camera_params(params) -> list[str]`

`BLCSCameraParams` への準拠を検証します。

**必須キー**:
- `center` (list of 3 floats)
- `R` (3x3 matrix)
- `f`, `cx`, `cy` (number)
- `w`, `h` (int, > 0)

---

### PLCS スキーマ

#### `validate_plcs_frame_batch(batch) -> list[str]`

`PLCSFrameBatch` TypedDict への準拠を検証します。

**期待されるスキーマ** (from `src/plcs/data/types.py`):

| キー | 形状 | 説明 |
|-----|------|------|
| `human_kp` | (34,) | フラット化された人体キーポイント（17×2）|
| `court_kp` | (40,) | フラット化されたコートキーポイント（20×2）|
| `human_vis` | (17,) | 人体キーポイント可視性 |
| `court_vis` | (20,) | コートキーポイント可視性 |
| `position` | (3,) | 正規化コート位置 |
| `rotation` | (2,) | プレーヤー向き [sin(yaw), cos(yaw)] |

**検証項目**:
- 必須キーの存在
- テンソル形状
- rotation が [-1, 1] 範囲
- 可視性マスクの値

#### `validate_plcs_sequence_batch(batch) -> list[str]`

`PLCSSequenceBatch` TypedDict への準拠を検証します。

**期待されるスキーマ**:

| キー | 形状 | 説明 |
|-----|------|------|
| `human_kp` | (T, 17, 2) | 時系列人体キーポイント |
| `court_kp` | (1, 20, 2) | 集約コートキーポイント |
| `human_vis` | (T, 17) | 時系列人体可視性 |
| `court_vis` | (1, 20) | 集約コート可視性 |
| `position` | (T, 3) | 時系列正規化位置 |
| `rotation` | (T, 2) | 時系列プレーヤー向き |

#### `validate_plcs_scene_meta(meta) -> list[str]`

`PLCSSceneMeta` への準拠を検証します。

**必須キー**:
- `scene_id`, `motion_source`, `motion_category` (str)
- `gender` ("male", "female", "neutral")
- `fps`, `num_frames`, `num_cameras_sampled`, `num_cameras` (int)
- `initial_position` (list)
- `initial_yaw` (number)

#### `validate_plcs_camera_params(params) -> list[str]`

BLCS と同じカメラパラメータスキーマを使用。

---

## 使用例

### サンプル検証の完全な例

```python
import pytest
from tests.e2e.validation import (
    validate_blcs_sample,
    validate_blcs_batch,
    validate_tensor_shape,
    validate_normalized_uv,
)

@pytest.mark.e2e
def test_dataset_sample_validation(blcs_dataset):
    """データセットサンプルがスキーマに準拠することを検証"""
    sample = blcs_dataset[0]
    
    # 完全なスキーマ検証
    errors = validate_blcs_sample(sample)
    assert not errors, f"Validation errors: {errors}"
    
    # 追加の個別検証
    uv_err = validate_normalized_uv(sample["ball_uv"], "ball_uv")
    assert uv_err is None, uv_err
```

### バッチ検証の例

```python
@pytest.mark.e2e
def test_dataloader_batch_validation(blcs_dataloader):
    """DataLoaderバッチがスキーマに準拠することを検証"""
    batch = next(iter(blcs_dataloader))
    
    errors = validate_blcs_batch(batch, batch_size=2)
    assert not errors, f"Batch validation errors: {errors}"
```

---

## 新規バリデータの追加

新しいスキーマバリデータを追加する手順：

1. `src/{task}/data/types.py` の TypedDict を確認
2. `tensor_validators.py` の基本関数を使用
3. `schema_validators.py` に `validate_{task}_{type}` 関数を追加
4. docstring に期待されるスキーマを記載
5. `tests/e2e/validation/__init__.py` にエクスポートを追加
