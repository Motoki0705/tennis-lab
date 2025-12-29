# WASB generate_dataset/download_videos

YouTubeからテニス映像をダウンロードするスクリプト。

## 概要

このスクリプトは、`urls.yaml` に定義されたYouTube URLからテニス映像をダウンロードします。yt-dlpを使用し、進捗追跡と再開機能をサポートします。

## コマンド例

```bash
# 全動画をダウンロード
uv run python -m src.wasb.scripts.generate_dataset.download_videos

# カスタム urls.yaml を指定
uv run python -m src.wasb.scripts.generate_dataset.download_videos urls_path=path/to/urls.yaml

# ステータス確認
uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=status

# 失敗したダウンロードをリセット
uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=reset_failed

# 全ダウンロードをリセット
uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=reset_all

# 特定のURLをリセット（再ダウンロード）
uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=reset_url reset_url="https://..."
```

## コンフィグ

エントリポイント: `src/wasb/configs/download_videos.yaml`

### モード

| モード | 説明 |
|--------|------|
| `download` | URLからダウンロード |
| `status` | ダウンロード状態を表示 |
| `reset_failed` | 失敗したダウンロードをリセット |
| `reset_all` | 全ダウンロードをリセット |
| `reset_url` | 特定のURLをリセット |

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mode` | `download` | 実行モード |
| `urls_path` | `data/tennis/raw/urls.yaml` | URL定義ファイル |
| `reset_url` | `null` | リセットするURL |
| `resume` | `true` | ダウンロードを再開するか |
| `verbose` | `true` | 詳細出力 |

## urls.yaml 形式

```yaml
videos:
  - url: "https://www.youtube.com/watch?v=..."
    name: "match1"                    # オプション: カスタムファイル名
    start: "00:05:00"                 # オプション: 開始時間
    end: "01:30:00"                   # オプション: 終了時間

  - url: "https://www.youtube.com/watch?v=..."
    name: "match2"

  - url: "https://youtu.be/..."       # 短縮URL も可
```

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          download_videos.py                                  │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │   urls.yaml     │──────▶│ VideoDownloader │──────▶│    yt-dlp           │  │
│  │                 │      │                 │      │                     │  │
│  │ - URL リスト    │      │ - 進捗管理      │      │ - ダウンロード      │  │
│  │ - 時間指定      │      │ - meta.json     │      │ - 時間抽出          │  │
│  │ - ファイル名    │      │ - 再開機能      │      │ - フォーマット変換  │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. urls.yaml を読み込み
2. meta.json から既存の進捗を確認
3. 未ダウンロードの URL について:
   a. yt-dlp でダウンロード
   b. (start/end 指定時) 時間範囲を抽出
   c. meta.json を更新
4. 失敗した場合はエラーを記録
```

## 出力構造

```
data/tennis/raw/
├── urls.yaml            # URL 定義
├── meta.json            # ダウンロード状態
└── videos/
    ├── match1.mp4
    ├── match2.mp4
    └── ...
```

## meta.json 形式

```json
{
  "version": "1.0",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-02T00:00:00",
  "urls_hash": "abc123...",
  "downloads": {
    "abc123def456": {
      "status": "completed",
      "url": "https://www.youtube.com/watch?v=...",
      "filename": "match1.mp4",
      "downloaded_at": "2024-01-01T12:00:00",
      "file_size": 1234567890
    },
    "def456ghi789": {
      "status": "failed",
      "url": "https://...",
      "error_message": "Video unavailable"
    }
  }
}
```

## ダウンロード状態

| 状態 | 説明 |
|------|------|
| `pending` | ダウンロード待ち |
| `in_progress` | ダウンロード中 |
| `completed` | 完了 |
| `failed` | 失敗 |

## 要件

- `yt-dlp` がインストールされている必要があります
- `pyyaml` も必要

```bash
pip install yt-dlp pyyaml
```

## 使用例

```bash
# 1. urls.yaml を作成
cat > data/tennis/raw/urls.yaml << EOF
videos:
  - url: "https://www.youtube.com/watch?v=XXXXX"
    name: "federer_vs_nadal"
    start: "00:10:00"
    end: "02:00:00"
EOF

# 2. ダウンロード実行
uv run python -m src.wasb.scripts.generate_dataset.download_videos

# 3. 状態確認
uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=status

# 4. 失敗したものを再試行
uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=reset_failed
uv run python -m src.wasb.scripts.generate_dataset.download_videos
```

## 注意事項

- YouTubeの利用規約を遵守してください
- 著作権のあるコンテンツのダウンロードには注意が必要です
- 研究・教育目的での使用を想定しています
