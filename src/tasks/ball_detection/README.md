# Ball Detection

Ball Detection は、テニス動画からボール位置ヒートマップを時系列で推定するための
タスク実装です。`src/tasks/ball_detection` は task-local な Hydra config、
データセット、STUNet モデル、学習ループ、半教師あり pseudo-label 生成をまとめています。

## 目的 / 想定入出力

- **入力**: RGB フレーム列
- **出力**: 各フレームに対するボール位置ヒートマップと、そこから得られる 2D ボール位置

### 入力形式

| モジュール | キー | 形状 | 説明 |
|-----------|------|------|------|
| `BallDetectionDataset` | `images` | `(B, T, 3, H, W)` | RGB 時系列入力 |
| `BallDetectionDataset` | `heatmaps` | `(B, T, Hh, Wh)` | 正規化座標から共有 utils で生成した教師ヒートマップ |
| `BallDetectionDataset` | `coords` | `(B, T, 2)` | 元画像ピクセル座標の GT |
| `BallDetectionDataset` | `visibility` | `(B, T)` | 各フレームのボール可視フラグ |
| `BallDetectionDataset` | `original_size` | `(B, 2)` | 元画像サイズ `(width, height)` |
| `BallDetectionDataset` | `heatmap_size` | `(B, 2)` | ヒートマップサイズ `(width, height)` |

### 出力形式

**モデル出力（`SpatioTemporalUNet.forward()`）:**

| モジュール | 出力 | 形状 | 型 | 説明 |
|-----------|------|------|-----|------|
| `SpatioTemporalUNet` | logits | `(B, 1, T, H/2, W/2)` | `torch.Tensor` | 各フレームのボール位置ロジット |

**学習・評価での扱い:**

- 学習時は `sigmoid(logits)` をヒートマップ確率として扱います。
- 評価時は `src/utils/data/heatmaps.py` の hard argmax で正規化座標を復元し、元画像座標へ戻して `precision / recall / f1 / mean_distance_px`
  を計算します。

## 実行コマンド

### 学習

```bash
# 既定設定で学習
python -m src.tasks.ball_detection.scripts.train

# 出力先を指定
python -m src.tasks.ball_detection.scripts.train \
    run.output_dir=outputs/ball_detection/stunet_custom

# バッチサイズを変更
python -m src.tasks.ball_detection.scripts.train \
    data.batch_size=8
```

### 半教師あり学習

```bash
# phase-based semi-supervised training
python -m src.tasks.ball_detection.scripts.train \
    training.semi_supervised.num_semi_phases=4 \
    training.semi_supervised.phase0_epochs=15 \
    training.semi_supervised.phase_epochs=15

# pseudo-label generation settings を上書き
python -m src.tasks.ball_detection.scripts.train \
    training.semi_supervised.num_semi_phases=4 \
    training.semi_supervised.pseudo_windows_per_video=128 \
    training.semi_supervised.pseudo_inference_batch_size=16
```

### 生動画ダウンロード

```bash
python -m src.tasks.ball_detection.scripts.download_videos
```

設定は `src/tasks/ball_detection/configs/downlaod.yaml` を使います。  
ダウンロード後の動画は `data/tennis/raw/videos/video_<n>.mp4` として保存され、
対応表は `data/tennis/raw/videos/summary.json` に書かれます。

### データ拡張の可視化

```bash
# train split の sample 0 を可視化
python -m src.tasks.ball_detection.scripts.visualize_augmentation \
    preview.sample_indices=[0]

# val split を複数サンプル出力
python -m src.tasks.ball_detection.scripts.visualize_augmentation \
    preview.split=val \
    preview.sample_indices=[0,1,2]
```

出力は `outputs/ball_detection/augmentation_preview/` に保存されます。  
上段が元シーケンス、下段が augmentation をすべて有効にしたシーケンスです。

### 推論可視化

```bash
# 既定の clip / checkpoint で GIF を生成
python -m src.tasks.ball_detection.scripts.visualize

# clip と出力先を上書き
python -m src.tasks.ball_detection.scripts.visualize \
    visualization.clip_dir=data/tennis/game1/Clip1 \
    visualization.save=assets/ball_detection/game1_clip1_prediction.gif
```

既定では `outputs/ball_detection/stunet/logs/version_2/checkpoints/ball-detection-epoch=18.ckpt`
を使い、clip 全体に sliding-window 推論を流して、重なった window の heatmap を
フレームごとに平均した上で GIF を生成します。

- 左パネル: RGB フレームに GT と予測位置を重ねた表示
- 右パネル: 予測 heatmap overlay
- GT は赤、しきい値以上の予測は緑で表示

![Ball detection visualization](../../../assets/ball_detection/game1_clip1_prediction.gif)

### コード・データのパッケージング

```bash
# code + data
python -m src.tasks.ball_detection.scripts.package

# code のみ
python -m src.tasks.ball_detection.scripts.package package_target=code

# data のみ
python -m src.tasks.ball_detection.scripts.package package_target=data
```

## 半教師あり pseudo-label フロー

`train.py` は `training.semi_supervised.num_semi_phases > 0` のとき、
phase 1 以降の開始前に pseudo-label を生成します。

1. `data/tennis/raw/videos/video_*.mp4` を列挙する
2. 各動画を `data/tennis/pseudo_label/cache/<video>/frames/` に逐次 decode する
3. chunk 単位で STUNet 推論を行い、重複 window の heatmap をフレームごとに平均する
4. confidence threshold と motion consistency で pseudo window を選別する
5. `data/tennis/pseudo_label/phase_XX/` に `Label.csv`、`manifest.jsonl`、`summary.json` を書く
6. 次 phase の train dataset で pseudo window を supervised window に追加する

主な設定は `src/tasks/ball_detection/configs/training/default.yaml` にあります。

## データ拡張

既定の supervised train 設定では、`src/tasks/ball_detection/configs/data/rgb_sequence.yaml`
の augmentation が有効です。

- camera rotation
- horizontal flip
- brightness gain
- contrast
- gamma
- gaussian noise
- gaussian blur

これらは `src/tasks/ball_detection/data/argumentation.py` の
`BallDetectionArgumentation` で時系列一貫性を保ったまま適用されます。

## モデルアーキテクチャ

### STUNet (`SpatioTemporalUNet`)

時空間 U-Net ベースのボール検出モデルです。

- 入力: `(B, C, T, H, W)`
- 出力: `(B, 1, T, H/2, W/2)`
- stem で空間方向を 1/2 に縮小
- encoder/decoder は 2D spatial block と 3D temporal block を組み合わせる構成

実装: `src/tasks/ball_detection/models/spatiotemporal_unet.py`

## 主要ファイル

```text
src/tasks/ball_detection/
├── configs/
│   ├── data/rgb_sequence.yaml
│   ├── model/stunet.yaml
│   ├── preview_heatmaps.yaml
│   ├── run/visualize.yaml
│   ├── train.yaml
│   ├── training/default.yaml
│   ├── visualize.yaml
│   ├── visualization/default.yaml
│   ├── visualize_augmentation.yaml
│   └── package.yaml
├── data/
│   ├── argumentation.py
│   ├── dataset.py
│   └── types.py
├── models/
│   └── spatiotemporal_unet.py
├── scripts/
│   ├── download_videos.py
│   ├── package.py
│   ├── preview_heatmaps.py
│   ├── train.py
│   ├── visualize.py
│   └── visualize_augmentation.py
├── visualization/
│   ├── orchestrator.py
│   └── rendering.py
└── training/
    ├── lightning_module.py
    ├── losses.py
    ├── metrics.py
    └── runner.py
```

## 注意点

- `visualize.py` は clip 全体に対して sliding-window 推論を行い、重なり heatmap を平均して GIF 化します。
- pseudo-label は `pseudo_label_root/phase_XX/` に phase ごとに保存されます。
- 生動画ベースの半教師あり学習では `data/tennis/raw/videos/` の準備が前提です。
- CUDA 上では `MaxPool3d` の backward に deterministic kernel が無い経路があるため、
  既定では `training.trainer.deterministic=warn` です。
