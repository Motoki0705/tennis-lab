# conftest.py

E2Eテストの共通設定とフィクスチャを定義するpytest設定ファイルです。

**ファイル**: `tests/e2e/conftest.py`

## 概要

このファイルは以下を提供します：

1. **pytest設定**: CUDAマーカーの登録と自動スキップ
2. **共通フィクスチャ**: 一時ディレクトリ、Hydraスクリプト実行ヘルパー
3. **型付きデコレータ**: mypy互換のフィクスチャ/マークデコレータ

## pytest フック

### `pytest_configure(config)`

`cuda` マーカーをpytestに登録します。

```python
@pytest.mark.cuda
def test_gpu_required():
    """CUDAが必要なテスト"""
    ...
```

### `pytest_collection_modifyitems(config, items)`

CUDAが利用不可の場合、`@pytest.mark.cuda` 付きテストを自動スキップします。

## フィクスチャ

### `schema_validators` (session)

スキーマ検証ユーティリティモジュールへのアクセスを提供します。

```python
def test_with_validators(schema_validators):
    errors = schema_validators.validate_blcs_sample(sample)
```

### `tmp_output_dir` (session)

E2Eテスト用の一時出力ディレクトリを作成します。

```python
def test_output(tmp_output_dir: Path):
    output_path = tmp_output_dir / "result.npz"
```

### `tmp_data_dir` (session)

E2Eテスト用の一時データディレクトリを作成します。

```python
def test_data(tmp_data_dir: Path):
    data_path = tmp_data_dir / "scene.npz"
```

### `run_hydra_script` (session)

Hydra設定を使用するスクリプトを実行するヘルパー関数を提供します。

**シグネチャ**:
```python
def run_hydra_script(
    module: str,
    extra_args: Sequence[str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]
```

**使用例**:
```python
def test_script(run_hydra_script, tmp_output_dir):
    result = run_hydra_script(
        "src.blcs.scripts.generate_dataset",
        extra_args=["sampling.per_from_cell_samples=5"],
    )
    assert result.returncode == 0
```

**自動設定**:
- `run.output_dir={tmp_output_dir}`
- `run.data_dir={tmp_data_dir}`

## ユーティリティ関数

### `typed_fixture(*args, **kwargs)`

mypy互換の型付きpytestフィクスチャデコレータ。

```python
@typed_fixture(scope="session")
def my_fixture() -> MyType:
    return MyType()
```

### `typed_mark(mark)`

mypy互換の型付きpytestマークデコレータ。

```python
e2e = typed_mark(pytest.mark.e2e)
cuda = typed_mark(pytest.mark.cuda)

@e2e
@cuda
def test_gpu_e2e():
    ...
```
