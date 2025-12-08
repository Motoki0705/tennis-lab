# テニスデータセット (`data/tennis/`)

WASB ベースのテニスボール検出モデル向けに整形したデータセットです。
`src/wasb/scripts/generate_game.py` で生動画から自動生成し、
`third_party/WASB-SBDT` の `Tennis` データセットと互換な形式をとります。

## 1. ディレクトリ構造

```text
data/tennis/
├── game1/
├── game2/
│   └── ...      # 各ゲーム（1 試合）
├── game21/
├── meta.json    # 生成状態のメタ情報
├── raw/         # 入力動画を置くディレクトリ（任意）
└── samples/     # プレビュー用動画（任意）
```

- **`gameX/`**: 1 本の試合動画をクリップに分割した結果。
- **`meta.json`**: 入力動画と `gameX` の対応、処理ステータス、クリップ数などを管理。
- **`samples/`**: `--generate-samples` で生成される、ボールを上書き描画したプレビュー動画。

### 1.1 ゲームディレクトリ (`data/tennis/gameX/`)

```text
data/tennis/game11/
├── Clip1/
├── Clip2/
│   └── ...
└── ClipK/
```

- 各 `ClipK/` が 1 つのラリー（あるいは連続区間）に対応します。

### 1.2 クリップディレクトリ (`data/tennis/gameX/ClipK/`)

```text
data/tennis/game11/Clip1/
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── Label.csv
```

- `frame_XXXX.jpg`: 元動画から切り出したフレーム画像。
- `Label.csv`: 各フレームのボール位置と可視性ラベル。

`Label.csv` の `file name` カラムは、`frame_0001.jpg` に対して `0001.jpg` のように
プレフィックスを除いた名前を持ちます。

## 2. Label.csv のフォーマット

`src/wasb/tennis_format.py` で定義されているフォーマットに従います。

### 2.1 カラム

```text
file name, visibility, x-coordinate, y-coordinate, status, score
```

- **`file name`**: フレームファイル名（例: `0001.jpg`）。
- **`visibility`**: 可視性フラグ（0/1/2）。
- **`x-coordinate`**: 画像座標系でのボール中心 x 座標（ピクセル）。
- **`y-coordinate`**: 同 y 座標（ピクセル）。
- **`status`**: 追加フラグ用（現状は 0 固定）。
- **`score`**: 検出スコア（信頼度）。補完点では 0.0。

### 2.2 可視性ラベル (`visibility`)

- **0 (missing)**: 未検出 / アノテーションなし。`x, y, score` は 0 扱い。
- **1 (detected)**: モデルが検出した点。`x, y` は検出位置、`score` は検出スコア。
- **2 (completed)**: 物理モデルや BiLSTM による補完点。`x, y` は補完位置、`score` は 0.0。

これらの仕様により、WASB-SBDT の `Tennis` ローダ（`third_party/WASB-SBDT/src/datasets/tennis.py`）
からそのまま読み込める形式になっています。
