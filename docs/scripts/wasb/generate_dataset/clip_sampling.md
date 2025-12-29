# WASB generate_dataset/clip_sampling

クリップのプレビュー生成と手動選別ワークフローをサポートするスクリプト。

## 概要

このスクリプトは、生成されたクリップのプレビュー動画を作成し、手動で選別した結果を適用するワークフローを提供します。データセットのキュレーション（品質管理）に使用します。

## コマンド例

```bash
# サンプルプレビュー動画を生成
uv run python -m src.wasb.scripts.generate_dataset.clip_sampling mode=generate_samples \
  output_dir=data/tennis generate_samples=[game11]

# 全ゲームのサンプルを生成
uv run python -m src.wasb.scripts.generate_dataset.clip_sampling mode=generate_samples \
  output_dir=data/tennis generate_samples=[all]

# 手動選別結果を適用（削除されたプレビューのクリップを除去）
uv run python -m src.wasb.scripts.generate_dataset.clip_sampling mode=apply_clip_selection \
  output_dir=data/tennis apply_clip_selection=[game11]

# FPS を変更してプレビュー生成
uv run python -m src.wasb.scripts.generate_dataset.clip_sampling mode=generate_samples fps=30 \
  generate_samples=[game11,game12]
```

## コンフィグ

エントリポイント: `src/wasb/configs/clip_sampling.yaml`

### モード

| モード | 説明 |
|--------|------|
| `generate_samples` | プレビュー動画を生成 |
| `apply_clip_selection` | 手動選別結果を適用 |

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mode` | `generate_samples` | 実行モード |
| `output_dir` | `data/tennis` | データセットディレクトリ |
| `fps` | `50` | プレビュー動画のFPS |
| `generate_samples` | `[]` | サンプル生成対象ゲーム (例: `["game11"]`, `["all"]`) |
| `apply_clip_selection` | `[]` | 選別適用対象ゲーム |
| `quiet` | `false` | 出力を抑制 |

## ワークフロー

### 1. サンプル生成

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  generate_samples モード                                                     │
│                                                                              │
│  data/tennis/game11/                 data/tennis/samples/game11/            │
│  ├── Clip1/                          ├── Clip_1.mp4                         │
│  │   ├── Label.csv          ──▶      ├── Clip_2.mp4                         │
│  │   └── frame_*.jpg                 ├── Clip_3.mp4                         │
│  ├── Clip2/                          └── ...                                │
│  └── ...                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

処理:
1. 各クリップのフレームと Label.csv を読み込み
2. ボール位置を赤丸でオーバーレイ
3. プレビュー動画として samples/ に保存
```

### 2. 手動選別

```
samples/game11/
├── Clip_1.mp4    ← 良いクリップ（残す）
├── Clip_2.mp4    ← 不要なクリップ（削除）
├── Clip_3.mp4    ← 良いクリップ（残す）
└── Clip_4.mp4    ← 不要なクリップ（削除）
```

不要なプレビュー動画を手動で削除します。

### 3. 選別適用

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  apply_clip_selection モード                                                 │
│                                                                              │
│  samples/game11/                     data/tennis/game11/                    │
│  ├── Clip_1.mp4  (残存)              ├── Clip1/  (元 Clip1)                 │
│  └── Clip_3.mp4  (残存)      ──▶     └── Clip2/  (元 Clip3, 番号再振り)     │
│                                                                              │
│  Clip_2.mp4, Clip_4.mp4 は削除されているため、                               │
│  対応するクリップディレクトリも削除される                                    │
└─────────────────────────────────────────────────────────────────────────────┘

処理:
1. samples/ に残っているプレビュー動画を確認
2. 対応するクリップのみを残す
3. クリップ番号を連番に振り直す
```

## 出力構造

```
data/tennis/
├── game11/
│   ├── Clip1/           # 選別後のクリップ（番号再振り）
│   ├── Clip2/
│   └── ...
└── samples/
    └── game11/
        ├── Clip_1.mp4   # プレビュー動画
        ├── Clip_3.mp4
        └── ...
```

## プレビュー動画の内容

- 元フレーム画像
- 検出されたボール位置に赤丸をオーバーレイ
- visibility が 0 以外の場合のみ表示

```python
# オーバーレイの実装例
if row.visibility != 0:
    cv2.circle(frame, (x, y), 8, (0, 0, 255), 2, cv2.LINE_AA)
```

## 使用例

```bash
# 1. 新しいゲームのクリップを生成（batch.py）
uv run python -m src.wasb.scripts.generate_dataset mode=batch

# 2. プレビュー動画を生成
uv run python -m src.wasb.scripts.generate_dataset.clip_sampling \
  mode=generate_samples generate_samples=[all]

# 3. samples/ ディレクトリを確認し、不要なプレビューを削除

# 4. 選別結果を適用
uv run python -m src.wasb.scripts.generate_dataset.clip_sampling \
  mode=apply_clip_selection apply_clip_selection=[all]
```

## 関連モジュール

- `src.wasb.tennis_format`: Label.csv の読み込み
