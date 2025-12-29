# 検証ユーティリティ

E2Eテストで使用するスキーマ検証およびテンソル検証ユーティリティの概要です。

## ファイル一覧

| ファイル | ドキュメント | 説明 |
|---------|-------------|------|
| `tensor_validators.py` | [tensor_validators.md](tensor_validators.md) | テンソル形状・dtype・値範囲の検証 |
| `schema_validators.py` | [schema_validators.md](schema_validators.md) | TypedDict/dataclassスキーマの検証 |

## 目的

検証ユーティリティは以下を確認します：

1. **テンソル形状**: 期待される次元と各次元のサイズ
2. **dtype**: float32, float64, bool 等の型
3. **値範囲**: [0, 1] や [-1, 1] 等の正規化範囲
4. **スキーマ準拠**: TypedDict で定義されたキーと型

## 使用パターン

### 個別検証

```python
from tests.e2e.validation import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_tensor_range,
)

# 形状検証（Noneは任意の値を許可）
err = validate_tensor_shape(tensor, (None, 2), "ball_uv")

# dtype検証
err = validate_tensor_dtype(tensor, [torch.float32, torch.float64], "positions")

# 値範囲検証
err = validate_tensor_range(tensor, 0.0, 1.0, "normalized_coords")
```

### スキーマ検証

```python
from tests.e2e.validation import validate_blcs_sample

errors = validate_blcs_sample(sample)
assert not errors, f"Validation errors: {errors}"
```

### エラー収集

```python
from tests.e2e.validation import collect_errors

errors = collect_errors(
    validate_tensor_shape(t1, (32, 3), "t1"),
    validate_tensor_shape(t2, (32, 2), "t2"),
)
# errors = ["t1: ..."] or []
```
