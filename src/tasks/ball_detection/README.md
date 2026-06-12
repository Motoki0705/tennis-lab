# Ball Detection

テニス動画のRGBフレーム列から、各フレームのボール位置ヒートマップを推定するタスクです。
複数ボールの教師データ、TrackNet形式データ、YouTubeから作成したデータセットに対応しています。

## 設計

データ読み込みは次の責任に分かれています。

- `TrackNetDataModule`
  - splitファイル、ディレクトリ構造、`Label.csv`を解釈する
  - 固定長の`ClipWindow`を作成する
- `YouTubeDataModule`
  - `TrackNetDataModule`を継承する
  - YouTubeデータセットのsplitエントリ基準だけを変更する
- `BallDetectionDataset`
  - データ形式に依存しない`ClipWindow`を受け取る
  - 画像読み込み、augmentation、heatmap生成、Tensor化を行う

新しいデータ形式を追加する場合は、DataModuleで`ClipWindow`へ変換します。
Dataset側のsample生成処理は共通で再利用できます。

## Sample契約

`T`はフレーム数、`K`は`data.max_instances`です。

| キー | Sample形状 | Batch形状 | 説明 |
|---|---:|---:|---|
| `images` | `(T, 3, H, W)` | `(B, T, 3, H, W)` | RGBフレーム列 |
| `heatmaps` | `(T, Hh, Wh)` | `(B, T, Hh, Wh)` | 全可視ボールを統合した教師heatmap |
| `coords` | `(T, K, 2)` | `(B, T, K, 2)` | 元画像ピクセル座標の複数ボールGT |
| `visibility` | `(T, K)` | `(B, T, K)` | 可視インスタンスmask |
| `original_size` | `(2,)` | `(B, 2)` | `(width, height)` |
| `heatmap_size` | `(2,)` | `(B, 2)` | `(width, height)` |

`coords`の未使用領域は`(0, 0)`、対応する`visibility`は`0`でpaddingされます。

## データ形式

### TrackNet

```text
data/tennis/
├── game1/
│   ├── Clip1/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── Label.csv
│   └── Clip2/
└── game2/
```

splitファイルには`data.data_dir`からの相対パスを書きます。

```text
game1
game2/Clip1
```

`Label.csv`の必須列は次の4つです。

```csv
file name,visibility,x-coordinate,y-coordinate
000000.jpg,1,320.5,180.0
000001.jpg,0,0,0
```

複数ボールを保持する場合は、同じ`file name`で複数行を記録します。
`instance id`、`ball state`、`role`も利用できます。`role=distractor`は学習対象外です。

### YouTube

```text
data/tennis/youtube/
├── videos/
├── frames/
│   └── video_000001/
│       ├── raw/
│       └── clip_000001/
│           ├── 000000.jpg
│           └── Label.csv
├── staging/
├── annotations/
│   ├── train.txt
│   └── val.txt
└── manifests/
```

学習時はHydraのdata configを切り替えます。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train \
    data=youtube_rgb_sequence
```

## 学習と評価

### 学習方式

現在の学習フローは教師あり学習のみです。

YouTubeデータセット作成時のモデル予測は、人手アノテーションの初期値を作るための
補助機能です。人手で確認してfinalizeしたデータは、通常の教師データとして学習します。

### 学習

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train
```

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train \
    model=conv_next_unet \
    data.batch_size=4
```

利用可能なモデルは次のとおりです。

- `stunet`
- `conv_next_unet`
- `dino_pseudo3d`

### 評価

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.eval \
    run.checkpoint_path=path/to/checkpoint.ckpt
```

複数の予測peakと複数のGTを抽出し、ハンガリアン法で対応付けて
`precision`、`recall`、`f1`、`mean_distance_px`を計算します。

## Preview

### Augmentation

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.preview_augmentation
```

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.preview_augmentation \
    preview.split=val \
    preview.sample_indices=[0,1,2]
```

PNG、sample metadata、`manifest.json`は次へ保存されます。

```text
outputs/ball_detection/augmentation_preview/
```

上段が元シーケンス、下段がaugmentationをすべて有効にしたシーケンスです。

### Heatmap

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.preview_heatmaps
```

### Prediction GIF

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.visualize \
    visualization.clip_dir=data/tennis/game1/Clip1 \
    visualization.save=outputs/ball_detection/prediction.gif
```

## YouTubeデータセット作成

YouTube関連のCLIは`scripts/youtube/`にまとめています。

### 1. 動画準備とフレーム抽出

```bash
.venv/bin/python -m \
    src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset
```

`configs/prepare_youtube_dataset.yaml`の`workflow.sources`を読み、
動画のdownload、H.264への変換、連続フレーム抽出を行います。

### 2. 候補clip選択

```bash
.venv/bin/python -m \
    src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset \
    workflow.video_id=video_000001 \
    workflow.mode=select
```

### 3. アノテーション初期値のモデル予測

```bash
.venv/bin/python -m \
    src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset \
    workflow.video_id=video_000001 \
    workflow.mode=predict
```

この予測は人手確認を効率化するための初期値であり、半教師あり学習へ直接投入される
疑似ラベルではありません。

### 4. 人手確認と確定

```bash
.venv/bin/python -m \
    src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball \
    annotate.video_id=video_000001
```

全フレームを確認してfinalizeすると、clip、`Label.csv`、splitファイルが
YouTube学習データセットへ書き出されます。

## 主要ファイル

```text
src/tasks/ball_detection/
├── generate_dataset/
│   ├── annotation_session.py
│   ├── candidate_workflow.py
│   └── __init__.py
├── configs/
│   ├── data/
│   │   ├── rgb_sequence.yaml
│   │   └── youtube_rgb_sequence.yaml
│   ├── model/
│   ├── preview_augmentation.yaml
│   ├── preview_heatmaps.yaml
│   └── train.yaml
├── data/
│   ├── augmentation.py
│   ├── dataset.py
│   ├── tracknet_datamodule.py
│   ├── types.py
│   └── youtube_datamodule.py
├── models/
├── scripts/
│   ├── eval.py
│   ├── preview_augmentation.py
│   ├── preview_heatmaps.py
│   ├── train.py
│   ├── visualize.py
│   └── youtube/
│       ├── annotate_youtube_ball.py
│       ├── clip_and_predict_youtube_dataset.py
│       └── prepare_youtube_dataset.py
├── training/
├── visualization/
```

## 注意点

- プロジェクトコマンドには`.venv/bin/python`を使用します。
- YouTubeのdownloadには`yt-dlp`、変換には`ffmpeg`が必要です。
- annotation UIとclip選択UIはOpenCVのGUIを使用します。
- GPUがない場合は、学習・推論設定のdeviceや`run.gpus`をCPU向けに変更してください。
