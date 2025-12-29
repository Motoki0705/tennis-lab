# WASB E2E テスト

WASB (Where's the Ball) のE2Eテスト概要です。

## テストファイル一覧

| ファイル | ドキュメント | 説明 |
|---------|-------------|------|
| `test_data_validation.py` | [test_data_validation.md](test_data_validation.md) | データセットのスキーマ検証 |
| `test_generate_dataset.py` | [test_generate_dataset.md](test_generate_dataset.md) | データセット生成スクリプト |
| `test_tools.py` | [test_tools.md](test_tools.md) | ツールスクリプト |
| `test_train.py` | [test_train.md](test_train.md) | 学習スクリプト |
| `test_visualize.py` | [test_visualize.md](test_visualize.md) | 可視化スクリプト |

## テスト実行

```bash
# WASBテストのみ実行
uv run pytest tests/e2e/wasb -v

# GPUテストをスキップ
uv run pytest tests/e2e/wasb -v -m "not cuda"

# 学習テスト（ドライランのみ）
uv run pytest tests/e2e/wasb/test_train.py -v -m "not cuda"
```

## モデルの種類

| スクリプト | モデル | 説明 |
|-----------|--------|------|
| `train.ball_detection` | DinoV3 FPN Heatmap | ヒートマップベースのボール検出 |
| `train.trajectory` | TrajectoryBiLSTM | 軌道補完（欠損フレームの補間） |
| `train.event_detection` | イベント検出器 | バウンス/ヒット等のイベント検出 |

## 使用するフィクスチャ

すべてのテストは `tests/e2e/fixtures/wasb_fixtures.py` を使用：

- `create_minimal_wasb_dataset`: 最小データセット生成
- `create_minimal_wasb_checkpoint`: ボール検出チェックポイント
- `create_minimal_trajectory_checkpoint`: 軌道補完チェックポイント
- `create_minimal_video`: 合成ビデオ生成

詳細は [fixtures/wasb_fixtures.md](../fixtures/wasb_fixtures.md) を参照。

## 注意事項

WASBテストの多くは複雑な依存関係のためスキップされています：

- 実際のWASBモデルにはDinoV3バックボーンが必要
- ビデオ処理には追加の設定が必要
- 詳細は各テストドキュメントを参照
