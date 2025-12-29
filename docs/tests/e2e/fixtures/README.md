# テストフィクスチャ

E2Eテストで使用するテストデータ生成ユーティリティの概要です。

## ファイル一覧

| ファイル | ドキュメント | 説明 |
|---------|-------------|------|
| `blcs_fixtures.py` | [blcs_fixtures.md](blcs_fixtures.md) | BLCSテストデータ生成 |
| `plcs_fixtures.py` | [plcs_fixtures.md](plcs_fixtures.md) | PLCSテストデータ生成 |
| `wasb_fixtures.py` | [wasb_fixtures.md](wasb_fixtures.md) | WASBテストデータ生成 |

## 目的

フィクスチャは以下の目的で使用されます：

1. **最小限のテストデータ生成**: 高速にテストを実行するための軽量データ
2. **再現性のあるデータ**: 決定論的またはシード固定のデータ生成
3. **正しいスキーマのデータ**: TypedDict/dataclass準拠のデータ構造

## 共通パターン

### データセット生成

```python
from tests.e2e.fixtures.{task}_fixtures import create_minimal_{task}_dataset

dataset_dir = create_minimal_{task}_dataset(tmp_path / "data", num_scenes=5)
```

### チェックポイント生成

```python
from tests.e2e.fixtures.{task}_fixtures import create_minimal_{task}_checkpoint

checkpoint_path = create_minimal_{task}_checkpoint(tmp_path / "model.ckpt")
```

### pytestフィクスチャとの組み合わせ

```python
import pytest
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_dataset

@pytest.fixture
def blcs_dataset_dir(tmp_path):
    return create_minimal_blcs_dataset(tmp_path / "blcs_data", num_scenes=5)

def test_something(blcs_dataset_dir):
    assert (blcs_dataset_dir / "scenes").exists()
```
