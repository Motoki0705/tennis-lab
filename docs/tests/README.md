# テストドキュメント

tennis-lab のテストスイート（E2Eテスト）に関するドキュメントです。

## テスト構造

```
tests/
├── e2e/                        # End-to-End テスト
│   ├── conftest.py            # 共通フィクスチャ・設定
│   ├── fixtures/              # テスト用データ生成ユーティリティ
│   │   ├── blcs_fixtures.py   # BLCSテストデータ生成
│   │   ├── plcs_fixtures.py   # PLCSテストデータ生成
│   │   └── wasb_fixtures.py   # WASBテストデータ生成
│   ├── validation/            # スキーマ・テンソル検証ユーティリティ
│   │   ├── schema_validators.py   # TypedDict/dataclassスキーマ検証
│   │   └── tensor_validators.py   # テンソル形状・dtype・値範囲検証
│   ├── blcs/                  # BLCS E2Eテスト
│   ├── plcs/                  # PLCS E2Eテスト
│   └── wasb/                  # WASB E2Eテスト
└── unit/                      # ユニットテスト（本ドキュメントの対象外）
```

## E2Eテストの目的

E2Eテストは、以下の観点からスクリプトとパイプラインの正確性を検証します：

1. **スクリプト実行**: Hydra設定を使用したスクリプトが正常終了すること
2. **出力生成**: 期待されるファイル（シーン、チェックポイント、可視化）が生成されること
3. **データ整合性**: 生成データがTypeDict/dataclassスキーマに準拠すること
4. **テンソル検証**: 形状、dtype、値範囲が正しいこと

## クイックスタート

### テスト実行

```bash
# 全E2Eテストを実行（CUDAなし環境ではGPUテストはスキップ）
uv run pytest tests/e2e -v

# 特定タスクのテストを実行
uv run pytest tests/e2e/blcs -v
uv run pytest tests/e2e/plcs -v
uv run pytest tests/e2e/wasb -v

# CUDAテストのみ実行
uv run pytest tests/e2e -v -m cuda

# CUDAテストを除外
uv run pytest tests/e2e -v -m "not cuda"
```

### マーカー

| マーカー | 説明 |
|---------|------|
| `@pytest.mark.e2e` | E2Eテストであることを示す |
| `@pytest.mark.cuda` | CUDAが必要（GPUがない場合スキップ） |

## 詳細ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [validation/README.md](validation/README.md) | スキーマ・テンソル検証ユーティリティの詳細 |
| [blcs/README.md](blcs/README.md) | BLCS E2Eテストの詳細 |
| [plcs/README.md](plcs/README.md) | PLCS E2Eテストの詳細 |
| [wasb/README.md](wasb/README.md) | WASB E2Eテストの詳細 |
| [fixtures/README.md](fixtures/README.md) | テストフィクスチャの使い方 |

## 共通フィクスチャ

`conftest.py` で定義される共通フィクスチャ：

| フィクスチャ | スコープ | 説明 |
|-------------|---------|------|
| `schema_validators` | session | スキーマ検証モジュールへのアクセス |
| `tmp_output_dir` | session | 一時出力ディレクトリ |
| `tmp_data_dir` | session | 一時データディレクトリ |
| `run_hydra_script` | session | Hydraスクリプト実行ヘルパー |

## テスト追加ガイドライン

新しいE2Eテストを追加する際は：

1. 適切なマーカー（`@pytest.mark.e2e`, `@pytest.mark.cuda`）を付与
2. `fixtures/` のヘルパー関数を使用してテストデータを生成
3. `validation/` のユーティリティを使用してデータを検証
4. タイムアウトを設定（`timeout=300` など）
5. エラーメッセージに stdout/stderr を含める
