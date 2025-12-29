# E2E テスト

End-to-End テストの概要とドキュメント一覧です。

## 目的

E2Eテストは、以下の観点からスクリプトとパイプラインの正確性を検証します：

1. **スクリプト実行**: Hydra設定を使用したスクリプトが正常終了すること
2. **出力生成**: 期待されるファイル（シーン、チェックポイント、可視化）が生成されること
3. **データ整合性**: 生成データがTypeDict/dataclassスキーマに準拠すること
4. **テンソル検証**: 形状、dtype、値範囲が正しいこと

## ディレクトリ構造

```
tests/e2e/
├── conftest.py            # 共通フィクスチャ・設定
├── fixtures/              # テスト用データ生成
│   ├── blcs_fixtures.py
│   ├── plcs_fixtures.py
│   └── wasb_fixtures.py
├── validation/            # 検証ユーティリティ
│   ├── schema_validators.py
│   └── tensor_validators.py
├── blcs/                  # BLCS テスト
│   ├── test_data_validation.py
│   ├── test_generate_dataset.py
│   ├── test_train.py
│   └── test_visualize.py
├── plcs/                  # PLCS テスト
│   ├── test_data_validation.py
│   ├── test_generate_dataset.py
│   ├── test_train.py
│   └── test_visualize.py
└── wasb/                  # WASB テスト
    ├── test_data_validation.py
    ├── test_generate_dataset.py
    ├── test_tools.py
    ├── test_train.py
    └── test_visualize.py
```

## ドキュメント一覧

### 共通

| ドキュメント | テストファイル | 説明 |
|-------------|---------------|------|
| [conftest.md](conftest.md) | `conftest.py` | 共通フィクスチャとpytest設定 |

### fixtures/

| ドキュメント | テストファイル | 説明 |
|-------------|---------------|------|
| [fixtures/README.md](fixtures/README.md) | - | フィクスチャ概要 |
| [fixtures/blcs_fixtures.md](fixtures/blcs_fixtures.md) | `blcs_fixtures.py` | BLCSテストデータ生成 |
| [fixtures/plcs_fixtures.md](fixtures/plcs_fixtures.md) | `plcs_fixtures.py` | PLCSテストデータ生成 |
| [fixtures/wasb_fixtures.md](fixtures/wasb_fixtures.md) | `wasb_fixtures.py` | WASBテストデータ生成 |

### validation/

| ドキュメント | テストファイル | 説明 |
|-------------|---------------|------|
| [validation/README.md](validation/README.md) | - | 検証ユーティリティ概要 |
| [validation/tensor_validators.md](validation/tensor_validators.md) | `tensor_validators.py` | テンソル検証 |
| [validation/schema_validators.md](validation/schema_validators.md) | `schema_validators.py` | スキーマ検証 |

### blcs/

| ドキュメント | テストファイル | 説明 |
|-------------|---------------|------|
| [blcs/README.md](blcs/README.md) | - | BLCS テスト概要 |
| [blcs/test_data_validation.md](blcs/test_data_validation.md) | `test_data_validation.py` | データ検証テスト |
| [blcs/test_generate_dataset.md](blcs/test_generate_dataset.md) | `test_generate_dataset.py` | データ生成テスト |
| [blcs/test_train.md](blcs/test_train.md) | `test_train.py` | 学習テスト |
| [blcs/test_visualize.md](blcs/test_visualize.md) | `test_visualize.py` | 可視化テスト |

### plcs/

| ドキュメント | テストファイル | 説明 |
|-------------|---------------|------|
| [plcs/README.md](plcs/README.md) | - | PLCS テスト概要 |
| [plcs/test_data_validation.md](plcs/test_data_validation.md) | `test_data_validation.py` | データ検証テスト |
| [plcs/test_generate_dataset.md](plcs/test_generate_dataset.md) | `test_generate_dataset.py` | データ生成テスト |
| [plcs/test_train.md](plcs/test_train.md) | `test_train.py` | 学習テスト |
| [plcs/test_visualize.md](plcs/test_visualize.md) | `test_visualize.py` | 可視化テスト |

### wasb/

| ドキュメント | テストファイル | 説明 |
|-------------|---------------|------|
| [wasb/README.md](wasb/README.md) | - | WASB テスト概要 |
| [wasb/test_data_validation.md](wasb/test_data_validation.md) | `test_data_validation.py` | データ検証テスト |
| [wasb/test_generate_dataset.md](wasb/test_generate_dataset.md) | `test_generate_dataset.py` | データ生成テスト |
| [wasb/test_tools.md](wasb/test_tools.md) | `test_tools.py` | ツールスクリプトテスト |
| [wasb/test_train.md](wasb/test_train.md) | `test_train.py` | 学習テスト |
| [wasb/test_visualize.md](wasb/test_visualize.md) | `test_visualize.py` | 可視化テスト |

## テスト実行

```bash
# 全E2Eテスト
uv run pytest tests/e2e -v

# 特定タスク
uv run pytest tests/e2e/blcs -v

# GPUテストをスキップ
uv run pytest tests/e2e -v -m "not cuda"

# 特定のテストファイル
uv run pytest tests/e2e/blcs/test_data_validation.py -v
```
