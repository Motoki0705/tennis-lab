# tensor_validators.py

テンソルの形状、dtype、値範囲を検証する低レベルユーティリティです。

**ファイル**: `tests/e2e/validation/tensor_validators.py`

## 関数一覧

| 関数 | 説明 |
|------|------|
| `validate_tensor_shape` | テンソル形状の検証 |
| `validate_tensor_dtype` | テンソルdtypeの検証 |
| `validate_tensor_range` | テンソル値範囲の検証 |
| `validate_normalized_uv` | 正規化UV座標の検証 |
| `validate_visibility_mask` | 可視性マスクの検証 |
| `validate_dict_has_keys` | 辞書の必須キー検証 |
| `collect_errors` | エラー収集ヘルパー |

---

## `validate_tensor_shape`

テンソル形状を検証します。動的次元には `None` を使用。

### シグネチャ

```python
def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int | None, ...],
    name: str,
) -> str | None
```

### パラメータ

| 名前 | 型 | 説明 |
|------|-----|------|
| `tensor` | Tensor | 検証対象のテンソル |
| `expected_shape` | tuple | 期待される形状（`None`は任意の値） |
| `name` | str | エラーメッセージ用の名前 |

### 戻り値

- エラーがあれば文字列
- なければ `None`

### 使用例

```python
# 固定形状
err = validate_tensor_shape(tensor, (32, 3), "positions")

# 動的次元（Tは任意）
err = validate_tensor_shape(tensor, (None, 2), "ball_uv")  # (T, 2)

# バッチ+時間が動的
err = validate_tensor_shape(tensor, (None, None, 3), "batch_positions")  # (B, T, 3)
```

---

## `validate_tensor_dtype`

テンソルのdtypeを検証します。

### シグネチャ

```python
def validate_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype | list[torch.dtype],
    name: str,
) -> str | None
```

### 使用例

```python
# 単一dtype
err = validate_tensor_dtype(tensor, torch.float32, "positions")

# 複数dtype許可
err = validate_tensor_dtype(tensor, [torch.float32, torch.float64], "positions")
```

---

## `validate_tensor_range`

テンソル値が指定範囲内かを検証します。

### シグネチャ

```python
def validate_tensor_range(
    tensor: torch.Tensor,
    min_val: float | None,
    max_val: float | None,
    name: str,
    *,
    allow_nan: bool = False,
) -> str | None
```

### チェック項目

- NaN の検出（`allow_nan=False` の場合）
- Inf の検出（常にエラー）
- 最小値・最大値の範囲

### 使用例

```python
# [0, 1] 範囲
err = validate_tensor_range(tensor, 0.0, 1.0, "normalized_coords")

# 下限のみ
err = validate_tensor_range(tensor, 0.0, None, "positive_values")

# NaN許可
err = validate_tensor_range(tensor, 0.0, 1.0, "with_missing", allow_nan=True)
```

---

## `validate_normalized_uv`

正規化UV座標（[0, 1]範囲）を検証します。

### シグネチャ

```python
def validate_normalized_uv(
    tensor: torch.Tensor,
    name: str,
    *,
    strict: bool = False,
) -> str | None
```

### 要件

- 最後の次元が2（UV座標）
- 値が [0, 1]（`strict=True`）または [0-0.01, 1+0.01]（`strict=False`）

### 使用例

```python
# 数値誤差を許容
err = validate_normalized_uv(tensor, "ball_uv")

# 厳密に [0, 1]
err = validate_normalized_uv(tensor, "ball_uv", strict=True)
```

---

## `validate_visibility_mask`

可視性マスクを検証します。

### シグネチャ

```python
def validate_visibility_mask(
    tensor: torch.Tensor,
    name: str,
) -> str | None
```

### 許可される値

- `torch.bool` 型
- 数値型で 0/1 のみ

### 使用例

```python
err = validate_visibility_mask(tensor, "ball_mask")
```

---

## `validate_dict_has_keys`

辞書に必須キーがあるかを検証します。

### シグネチャ

```python
def validate_dict_has_keys(
    data: dict[str, Any],
    required_keys: list[str],
    name: str,
) -> list[str]
```

### 戻り値

欠落キーのエラーメッセージリスト（空なら成功）

### 使用例

```python
errors = validate_dict_has_keys(
    sample,
    ["ball_uv", "ball_mask", "court_kp"],
    "BLCSSample"
)
```

---

## `collect_errors`

複数の検証結果から非Noneエラーを収集します。

### シグネチャ

```python
def collect_errors(*errors: str | None) -> list[str]
```

### 使用例

```python
errors = collect_errors(
    validate_tensor_shape(t1, (32, 3), "t1"),
    validate_tensor_shape(t2, (32, 2), "t2"),
    validate_tensor_dtype(t3, torch.float32, "t3"),
)
# errors = ["t1: dim 1 expected 3, got 2"] など
```
