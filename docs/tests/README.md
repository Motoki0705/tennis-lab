# テストドキュメント

tennis-lab のテストスイート（E2Eテスト）に関するドキュメントです。

## テスト構造

```
tests/
├── e2e/                        # End-to-End テスト
│   ├── conftest.py            # 共通フィクスチャ・設定
│   ├── fixtures/              # テスト用データ生成ユーティリティ
│   ├── validation/            # スキーマ・テンソル検証ユーティリティ
│   ├── blcs/                  # BLCS E2Eテスト
│   ├── plcs/                  # PLCS E2Eテスト
│   └── wasb/                  # WASB E2Eテスト
└── unit/                      # ユニットテスト（本ドキュメントの対象外）
```

## ドキュメント構造

このドキュメントは `tests/` と同じ構造で構成されています：

- [e2e/README.md](e2e/README.md) - E2Eテストの概要とまとめ

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
