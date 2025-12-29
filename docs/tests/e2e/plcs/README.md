# PLCS E2E テスト

PLCS (Player Location from Court Skeleton) のE2Eテスト概要です。

## テストファイル一覧

| ファイル | ドキュメント | 説明 |
|---------|-------------|------|
| `test_data_validation.py` | [test_data_validation.md](test_data_validation.md) | 生成データのスキーマ検証 |
| `test_generate_dataset.py` | [test_generate_dataset.md](test_generate_dataset.md) | データセット生成スクリプト |
| `test_train.py` | [test_train.md](test_train.md) | 学習スクリプト |
| `test_visualize.py` | [test_visualize.md](test_visualize.md) | 可視化スクリプト |

## テスト実行

```bash
# PLCSテストのみ実行
uv run pytest tests/e2e/plcs -v

# GPUテストをスキップ
uv run pytest tests/e2e/plcs -v -m "not cuda"

# シーケンスモデル関連のテスト
uv run pytest tests/e2e/plcs -v -k "sequence"
```

## モデルの種類

| スクリプト | モデル | 説明 |
|-----------|--------|------|
| `train` | `PLCSModel` | 単一フレーム入力で位置・回転を予測 |
| `train_sequence` | `PLCSSequenceModel` | 時系列入力で位置・回転を予測 |

## 使用するフィクスチャ

すべてのテストは `tests/e2e/fixtures/plcs_fixtures.py` を使用：

- `create_minimal_plcs_dataset`: 最小データセット生成
- `create_minimal_plcs_checkpoint`: フレームモデルチェックポイント
- `create_minimal_plcs_sequence_checkpoint`: シーケンスモデルチェックポイント
- `make_minimal_plcs_scene`: 単一シーン生成

詳細は [fixtures/plcs_fixtures.md](../fixtures/plcs_fixtures.md) を参照。
