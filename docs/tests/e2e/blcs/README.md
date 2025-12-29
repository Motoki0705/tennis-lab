# BLCS E2E テスト

BLCS (Ball Location from Court Skeleton) のE2Eテスト概要です。

## テストファイル一覧

| ファイル | ドキュメント | 説明 |
|---------|-------------|------|
| `test_data_validation.py` | [test_data_validation.md](test_data_validation.md) | 生成データのスキーマ検証 |
| `test_generate_dataset.py` | [test_generate_dataset.md](test_generate_dataset.md) | データセット生成スクリプト |
| `test_train.py` | [test_train.md](test_train.md) | 学習スクリプト |
| `test_visualize.py` | [test_visualize.md](test_visualize.md) | 可視化スクリプト |

## テスト実行

```bash
# BLCSテストのみ実行
uv run pytest tests/e2e/blcs -v

# GPUテストをスキップ
uv run pytest tests/e2e/blcs -v -m "not cuda"

# 特定のテストファイル
uv run pytest tests/e2e/blcs/test_data_validation.py -v
```

## 使用するフィクスチャ

すべてのテストは `tests/e2e/fixtures/blcs_fixtures.py` を使用：

- `create_minimal_blcs_dataset`: 最小データセット生成
- `create_minimal_blcs_checkpoint`: モデルチェックポイント生成
- `make_minimal_blcs_scene`: 単一シーン生成

詳細は [fixtures/blcs_fixtures.md](../fixtures/blcs_fixtures.md) を参照。
