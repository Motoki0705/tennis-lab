# schema_validators.py

TypedDict/dataclassスキーマに対する高レベル検証ユーティリティです。

**ファイル**: `tests/e2e/validation/schema_validators.py`

## 関数一覧

### BLCS

| 関数 | 説明 |
|------|------|
| `validate_blcs_sample` | BLCSSample スキーマ検証 |
| `validate_blcs_batch` | BLCSBatch スキーマ検証 |
| `validate_blcs_scene_meta` | BLCSSceneMeta スキーマ検証 |
| `validate_blcs_camera_params` | BLCSCameraParams スキーマ検証 |

### PLCS

| 関数 | 説明 |
|------|------|
| `validate_plcs_frame_batch` | PLCSFrameBatch スキーマ検証 |
| `validate_plcs_sequence_batch` | PLCSSequenceBatch スキーマ検証 |
| `validate_plcs_scene_meta` | PLCSSceneMeta スキーマ検証 |
| `validate_plcs_camera_params` | PLCSCameraParams スキーマ検証 |

---

## BLCS スキーマ

### `validate_blcs_sample`

`BLCSSample` TypedDict への準拠を検証します。

#### シグネチャ

```python
def validate_blcs_sample(sample: dict[str, Any]) -> list[str]
```

#### 期待されるスキーマ

| キー | 形状 | 説明 |
|-----|------|------|
| `ball_uv` | (T, 2) | 正規化UVでのボール2D軌道 |
| `ball_mask` | (T,) | ボール可視性マスク |
| `court_kp` | (20, 2) | コート2Dキーポイント（正規化UV） |
| `court_vis` | (20,) | コートキーポイント可視性 |
| `position_3d` | (T, 3) | グラウンドトゥルース3D軌道 |
| `velocity_3d` | (T, 3) | 3D速度ベクトル |
| `seq_len` | scalar | 実際のシーケンス長 |

#### 検証項目

- 必須キーの存在
- テンソル形状（`seq_len` に基づく）
- dtype（float32/float64）
- UV座標の正規化（[0, 1]）
- 可視性マスクの値（0/1）

#### 使用例

```python
from tests.e2e.validation import validate_blcs_sample

errors = validate_blcs_sample(sample)
assert not errors, f"Validation errors: {errors}"
```

---

### `validate_blcs_batch`

`BLCSBatch` TypedDict への準拠を検証します。

#### シグネチャ

```python
def validate_blcs_batch(batch: dict[str, Any], batch_size: int | None = None) -> list[str]
```

#### 期待されるスキーマ

| キー | 形状 | 説明 |
|-----|------|------|
| `ball_uv` | (B, T_max, 2) | パディング済みボール軌道 |
| `ball_mask` | (B, T_max) | パディング済み可視性マスク |
| `court_kp` | (B, 20, 2) | コートキーポイント |
| `court_vis` | (B, 20) | コートキーポイント可視性 |
| `position_3d` | (B, T_max, 3) | パディング済みグラウンドトゥルース |
| `velocity_3d` | (B, T_max, 3) | パディング済み速度 |
| `seq_len` | (B,) | 各サンプルの実際のシーケンス長 |

---

### `validate_blcs_scene_meta`

`BLCSSceneMeta` への準拠を検証します。

#### シグネチャ

```python
def validate_blcs_scene_meta(meta: dict[str, Any]) -> list[str]
```

#### 必須キー

| キー | 型 | 制約 |
|-----|-----|------|
| `scene_id` | str | - |
| `from_cell` | int | 0-11 |
| `from_side` | str | "near" or "far" |
| `category` | str | - |
| `to_cell` | int | - |
| `t_net`, `t_fence`, `t_bounce1`, `t_bounce2` | int | - |
| `fps_out`, `sim_fps`, `num_frames` | int | > 0 |
| `num_cameras_sampled`, `num_cameras` | int | - |

---

### `validate_blcs_camera_params`

`BLCSCameraParams` への準拠を検証します。

#### 必須キー

| キー | 型 | 説明 |
|-----|-----|------|
| `center` | list[float] | 長さ3 |
| `R` | list[list[float]] | 3x3行列 |
| `f`, `cx`, `cy` | number | 焦点距離、主点 |
| `w`, `h` | int | 画像サイズ（> 0） |

---

## PLCS スキーマ

### `validate_plcs_frame_batch`

`PLCSFrameBatch` TypedDict への準拠を検証します。

#### シグネチャ

```python
def validate_plcs_frame_batch(batch: dict[str, Any]) -> list[str]
```

#### 期待されるスキーマ

| キー | 形状 | 説明 |
|-----|------|------|
| `human_kp` | (34,) | フラット化された人体キーポイント（17×2）|
| `court_kp` | (40,) | フラット化されたコートキーポイント（20×2）|
| `human_vis` | (17,) | 人体キーポイント可視性 |
| `court_vis` | (20,) | コートキーポイント可視性 |
| `position` | (3,) | 正規化コート位置 |
| `rotation` | (2,) | プレーヤー向き [sin(yaw), cos(yaw)] |

#### 検証項目

- 必須キーの存在
- テンソル形状
- rotation が [-1, 1] 範囲
- 可視性マスクの値

---

### `validate_plcs_sequence_batch`

`PLCSSequenceBatch` TypedDict への準拠を検証します。

#### 期待されるスキーマ

| キー | 形状 | 説明 |
|-----|------|------|
| `human_kp` | (T, 17, 2) | 時系列人体キーポイント |
| `court_kp` | (1, 20, 2) | 集約コートキーポイント |
| `human_vis` | (T, 17) | 時系列人体可視性 |
| `court_vis` | (1, 20) | 集約コート可視性 |
| `position` | (T, 3) | 時系列正規化位置 |
| `rotation` | (T, 2) | 時系列プレーヤー向き |

---

### `validate_plcs_scene_meta`

`PLCSSceneMeta` への準拠を検証します。

#### 必須キー

| キー | 型 | 制約 |
|-----|-----|------|
| `scene_id` | str | - |
| `motion_source` | str | - |
| `motion_category` | str | - |
| `gender` | str | "male", "female", "neutral" |
| `fps`, `num_frames` | int | - |
| `num_cameras_sampled`, `num_cameras` | int | - |
| `initial_position` | list | - |
| `initial_yaw` | number | - |

---

### `validate_plcs_camera_params`

BLCS と同じカメラパラメータスキーマを使用します。

```python
def validate_plcs_camera_params(params: dict[str, Any]) -> list[str]:
    return validate_blcs_camera_params(params)
```
