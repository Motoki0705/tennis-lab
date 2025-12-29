# WASB generate_dataset

テニス映像からWASBデータセット（ボール位置アノテーション）を生成するスクリプト群。

## 概要

このスクリプト群は、生のテニス映像から学習用データセットを生成するパイプラインを提供します。事前学習済みWASBモデルを使用してボール位置を自動アノテーションし、tennis-lab形式のデータセットを作成します。

## メインエントリポイント

```bash
uv run python -m src.wasb.scripts.generate_dataset
```

これは `batch.py` の `main()` を呼び出します。

## コマンド例

```bash
# バッチモード: 指定ディレクトリの全動画を処理
uv run python -m src.wasb.scripts.generate_dataset mode=batch video_dir=data/tennis/raw

# ステータス確認
uv run python -m src.wasb.scripts.generate_dataset mode=status output_dir=data/tennis

# 失敗した動画をリセット
uv run python -m src.wasb.scripts.generate_dataset mode=reset_failed output_dir=data/tennis

# 特定の動画をリセット
uv run python -m src.wasb.scripts.generate_dataset mode=reset_video reset_video=[match1.mp4,match2.mp4]

# モデルとデバイスを指定
uv run python -m src.wasb.scripts.generate_dataset model=hrcnet device=cuda
```

## コンフィグ

エントリポイント: `src/wasb/configs/generate_dataset.yaml`

### モード

| モード | 説明 |
|--------|------|
| `batch` | 動画ディレクトリ内の全動画を処理 |
| `status` | 現在の処理状態を表示 |
| `reset_failed` | 失敗した動画のステータスをリセット |
| `reset_all` | 全動画のステータスをリセット |
| `reset_video` | 指定動画のステータスをリセット |

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mode` | `batch` | 実行モード |
| `video_dir` | `data/tennis/raw` | 入力動画ディレクトリ |
| `output_dir` | `data/tennis` | 出力ディレクトリ |
| `resume` | `true` | 処理を再開するか |
| `no_resume` | `false` | meta.json を無視して最初から |
| `start_game_id` | `null` | 開始ゲームID（自動検出） |
| `checkpoint` | `third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar` | モデルチェックポイント |
| `model` | `wasb` | モデル名 (wasb/hrcnet) |
| `device` | `cpu` | デバイス |
| `max_frames` | `null` | 動画あたりの最大フレーム数（テスト用） |
| `quiet` | `false` | 出力を抑制 |

### pipeline (パイプライン設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `score_threshold` | `0.5` | 検出スコア閾値 |
| `min_clip_length` | `30` | 最小クリップ長 |
| `min_detection_rate` | `0.5` | 最小検出率 |
| `max_gap` | `10` | クリップ分割の最大ギャップ |
| `clip_padding` | `5` | クリップのパディング |
| `batch_size` | `500` | バッチサイズ |
| `frame_format` | `frame_{:04d}.jpg` | フレームファイル名形式 |
| `jpeg_quality` | `95` | JPEG品質 |
| `use_completion` | `true` | 軌道補完を使用 |
| `completion_method` | `hybrid` | 補完方法 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              batch.py                                        │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │   VideoReader   │──────▶│   WASB/HRCNet   │──────▶│ AnnotationPipeline  │  │
│  │                 │      │   Predictor     │      │                     │  │
│  │ - フレーム抽出  │      │ - ボール検出    │      │ - クリップ分割      │  │
│  │ - バッチ処理    │      │ - スコア付け    │      │ - 軌道補完          │  │
│  │                 │      │                 │      │ - Label.csv 生成    │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
│                                                               │              │
│                                                               ▼              │
│                                                    ┌─────────────────────┐  │
│                                                    │   OutputWriter      │  │
│                                                    │                     │  │
│                                                    │ - game{N}/Clip{M}/  │  │
│                                                    │ - frame_XXXX.jpg    │  │
│                                                    │ - Label.csv         │  │
│                                                    │ - meta.json         │  │
│                                                    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. video_dir から動画ファイルを列挙
2. meta.json で処理状態を管理（resume サポート）
3. 各動画について:
   a. フレームを抽出
   b. WASB/HRCNet でボール位置を検出
   c. 検出結果を元にクリップに分割
   d. (use_completion=true) 軌道補完を適用
   e. game{N}/Clip{M}/ 構造で出力
4. meta.json を更新
```

## 出力構造

```
data/tennis/
├── meta.json                # 処理状態
├── game11/
│   ├── Clip1/
│   │   ├── Label.csv        # アノテーション
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   ├── Clip2/
│   │   └── ...
│   └── ...
├── game12/
│   └── ...
└── samples/                 # (clip_sampling で生成)
    └── ...
```

## Label.csv 形式

```csv
Frame,Visibility,X,Y,Status
0001.jpg,1,640.5,360.2,0
0002.jpg,1,650.3,355.8,0
0003.jpg,2,660.1,351.5,1
...
```

| 列 | 説明 |
|----|------|
| `Frame` | フレームファイル名 |
| `Visibility` | 0=不可視, 1=検出, 2=補完 |
| `X`, `Y` | ボールのピクセル座標 |
| `Status` | 0=なし, 1=ショット, 2=バウンス |

## meta.json 形式

```json
{
  "version": "1.0",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-02T00:00:00",
  "config": { ... },
  "videos": {
    "match1.mp4": {
      "status": "completed",
      "output_game": "game11",
      "num_clips": 25,
      "processed_at": "2024-01-01T12:00:00",
      "file_hash": "abc123..."
    },
    "match2.mp4": {
      "status": "pending",
      "output_game": "game12"
    }
  },
  "next_game_id": 13
}
```

## 関連モジュール

- `src.wasb.pipeline.AnnotationPipeline`: アノテーションパイプライン
- `src.wasb.inference.WASBPredictor`: WASB 推論
- `src.wasb.inference.HRCNetWASBPredictor`: HRCNet 推論
- `src.wasb.tennis_format`: Tennis 形式のI/O
