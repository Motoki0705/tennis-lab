# Ball Detection

> テニス映像のRGBフレーム列から、各フレームのボール位置を推定する時系列ボール検出タスク。

## 概要

複数フレームの時空間情報を統合し、高速・小物体であるテニスボールのヒートマップと座標・信頼度を出力します。

| 項目 | 内容 |
|---|---|
| 入力 | RGBフレーム列 `(B, T, 3, H, W)`（T=8、H=288、W=512 がデフォルト） |
| 出力 | 正規化ボール座標 `(B, T, 2)` + 信頼度 `(B, T)` + ヒートマップ（任意） |
| モデル | `stunet` / `conv_next_unet` / `dino_pseudo3d` |
| データ | TrackNet形式 + YouTube自作（手動アノテーション） |
| 役割 | 認識パイプラインの上流。2D観測を [BLCS](../blcs/) へ供給 |

## Inference

```python
import torch
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor

predictor = BallDetectionPredictor.load_from_checkpoint(
    "outputs/ball_detection/stunet/logs/version_0/checkpoints/last.ckpt",
    device="cuda",
)

# (B, T, 3, H, W), float32, 値域 [0, 1]
images = torch.rand(1, 8, 3, 288, 512)
result = predictor.predict(images, return_heatmaps=True)
coords = result["coords"]          # (1, 8, 2) 正規化座標
visibility = result["visibility"]  # (1, 8)
```

返り値は全テンソルが CPU 上の `dict[str, Tensor]`。

| キー | 形状 | dtype | 条件 | 説明 |
|---|---|---|---|---|
| `coords` | `(B, T, 2)` | float32 | 常時 | ピーク位置の正規化座標 `(x, y) ∈ [0, 1]` |
| `visibility` | `(B, T)` | float32 | 常時 | フレームごとのピーク信頼度 |
| `heatmaps` | `(B, T, H, W)` | float32 | `return_heatmaps=True` | sigmoid後の確率ヒートマップ（解像度はモデル依存、[モデル](#モデル)参照） |

`load_from_checkpoint` は単一チェックポイントのみ受け付けます（複数指定は `ValueError`）。

## データ

### TrackNet形式

```text
data/tennis/
└── game1/
    ├── Clip1/              # "Clip" / "clip_" で始まるディレクトリ
    │   ├── 000000.jpg      # 元解像度のフレーム
    │   └── Label.csv
    └── Clip2/
```

`Label.csv` の必須列は `file name, visibility, x-coordinate, y-coordinate`。任意列 `instance id` / `role` / `ball state` も解釈され、`role=distractor` は学習対象外です。複数ボールは同一 `file name` の複数行で表します。splitファイルには `data.data_dir` からの相対パスを記述します。

### YouTube形式

`YouTubeDataModule` は `TrackNetDataModule` を継承し、エントリ解決のみ変更します。splitは `data/tennis/youtube/annotations/{train,val,test}.txt`、パスは `data/tennis` からの相対（例 `youtube/frames/video_000001/clip_000001`）です。学習時は `data=youtube_rgb_sequence` を指定します。

### Sample契約（学習データ）

`T = model.num_frames`（=8）、`K = data.max_instances`（=16）。

| キー | 形状 | 説明 |
|---|---|---|
| `images` | `(T, 3, 288, 512)` | 正規化済みRGB |
| `heatmaps` | `(T, 288, 512)` | 全可視ボールを統合したガウス教師heatmap |
| `coords` | `(T, K, 2)` | 元解像度ピクセル座標の複数ボールGT |
| `visibility` | `(T, K)` | 可視インスタンスmask（未使用領域は座標 `(0,0)`・vis `0`） |
| `original_size` / `heatmap_size` | `(2,)` | `(width, height)` |

## データセット生成（YouTube）

YouTube動画から学習データを作る3ステップ。CLIは `scripts/youtube/` 配下です。

```bash
# 1. 動画download → H.264変換 → フレーム抽出（configs/prepare_youtube_dataset.yaml）
.venv/bin/python -m src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset

# 2a. 候補clipをGUIで選択
.venv/bin/python -m src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset \
    workflow.video_id=video_000001 workflow.mode=select
# 2b. 学習済みモデルでアノテーション初期値を予測
.venv/bin/python -m src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset \
    workflow.video_id=video_000001 workflow.mode=predict

# 3. OpenCV UIで人手確認・確定 → clip/Label.csv/splitを書き出し
.venv/bin/python -m src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball \
    annotate.video_id=video_000001
```

`predict` の出力は人手確認を効率化する初期値であり、半教師あり学習へ直接投入する疑似ラベルではありません。確定後のデータは通常の教師データとして扱います。

## 学習

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train
.venv/bin/python -m src.tasks.ball_detection.scripts.train model=conv_next_unet data.batch_size=4
```

デフォルトは `model=stunet`、`data=rgb_sequence`。`configs/training/default.yaml` は `max_epochs=20`、`lr=1e-4`、`precision=bf16-mixed`、`val/loss` 監視のEarlyStopping（patience=5）。

### 評価

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.eval \
    run.checkpoint_path=path/to/checkpoint.ckpt evaluation.splits=[val,test]
```

複数の予測ピークと複数GTをハンガリアン法で対応付け（距離閾値4.0px）、`precision` / `recall` / `f1` / `mean_distance_px` を `outputs/ball_detection/eval/` に出力します。

## 可視化

```bash
# ckptでクリップを推論し、予測GIFを生成
.venv/bin/python -m src.tasks.ball_detection.scripts.visualize \
    visualization.clip_dir=data/tennis/game1/Clip1 \
    visualization.checkpoint=path/to/checkpoint.ckpt \
    visualization.save=assets/ball_detection/prediction.gif

# データ確認用プレビュー（ckpt不要）
.venv/bin/python -m src.tasks.ball_detection.scripts.preview_heatmaps
.venv/bin/python -m src.tasks.ball_detection.scripts.preview_augmentation
```

## モデル

| 名前 | 入力モード / ch | 出力解像度 | 実装 |
|---|---|---|---|
| `stunet` | mdd / 2ch | H/2 × W/2 | `models/spatiotemporal_unet.py`（2D+3D Conv U-Net、T≥8 必須） |
| `conv_next_unet` | mdd / 2ch | H/4 × W/4 | `models/conv_next_unet.py`（ConvNeXt + 因子化Conv3d） |
| `dino_pseudo3d` | rgb / 3ch | フル解像度 | `models/dino_pseudo3d.py`（DINO backbone + Pseudo-3Dデコーダ） |

`mdd`（Motion Direction Decomposition）はRGBの輝度差分を明暗2chに分解する入力（`models/input_adapter.py`）。`rgb` はRGBをそのまま渡します。

## 補足

- 実行は `.venv/bin/python -m ...`。GPUが無い場合は `run.gpus` や device設定をCPU向けに変更します。
- YouTube生成は `yt-dlp`（download）・`ffmpeg`（H.264変換）に依存し、clip選択／アノテーションUIはOpenCVのGUIを使用します。
