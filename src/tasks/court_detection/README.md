# Court Detection

> テニス映像からコート構造を検出するタスク。キーポイント（`kp`）・セグメンテーション（`seg`）・ライン（`line`）の3タスクを統一実装で扱います。

## 概要

`encoder + decoder + 1×1 conv` の共通モデルで3タスクを学習し、`data` / `loss` config の切り替えで用途を選びます。

| 項目 | 内容 |
|---|---|
| 入力 | RGB画像（任意サイズ、内部で短辺リサイズ） |
| 出力（`kp`） | コートキーポイント座標 `(14, 2)` ピクセル + ヒートマップ（任意） |
| タスク | `kp`（14点）/ `seg`（7クラス）/ `line`（1クラス） |
| モデル | デフォルトEncoder+FPN / Swin-L(DINO)+FPN・UNet / DINOv3+DETR（`seg`専用） |
| データ | 手動アノテーションJSON |
| 役割 | 認識パイプラインの上流。コートKPを [BLCS](../blcs/) / [PLCS](../plcs/) へ供給 |

> **注**: 学習モデルが出力するキーポイントは index 0〜13 の **14点**（`configs/model/court_kp.yaml: num_classes=14`）。ネットポスト等を含む20点仕様（`NUM_COURT_KP=20`）はアノテーション用の正準定義で、現行モデルは出力しません。

## Inference

```python
import numpy as np
from PIL import Image
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor

predictor = CourtKeypointPredictor.load_from_checkpoint(
    "outputs/court_detection/kp/logs/version_0/checkpoints/last.ckpt",
    device="cpu",
)

image = np.array(Image.open("data/court/images/sample.jpg").convert("RGB"))
result = predictor.predict(image, return_heatmaps=False)
keypoints = result["keypoints"]  # (14, 2) 元画像ピクセル座標 [[x0, y0], ...]
```

入力は `np.ndarray` / `PIL.Image` / `Tensor` のいずれも可。返り値は全テンソルが CPU 上の `dict[str, Tensor]`。

| キー | 形状 | dtype | 条件 | 説明 |
|---|---|---|---|---|
| `keypoints` | `(K, 2)`（K=14） | float32 | 常時 | 元画像スケールのピクセル座標 `(x, y)` |
| `heatmaps` | `(K, H, W)` | float32 | `return_heatmaps=True` | 前処理後解像度のヒートマップ |

`short_side`（短辺リサイズ長）はckpt内 `config.data.augmentation.val_short_side`（デフォルト640）から自動取得します。

## キーポイント仕様（CourtKP20）

正準定義は `src/utils/schema/court.py` の `COURT_KP_NAMES` / `COURT_KP_IDX`。プロジェクト全体（CourtDetection / PLCS / BLCS / Rendering）がこのindexを共有します。学習モデルが出力するのは下表の **0〜13**。

| Index | Name | Index | Name |
|---|---|---|---|
| 0 | far_doubles_left | 7 | near_singles_right |
| 1 | far_doubles_right | 8 | far_service_left |
| 2 | near_doubles_left | 9 | far_service_right |
| 3 | near_doubles_right | 10 | near_service_left |
| 4 | far_singles_left | 11 | near_service_right |
| 5 | near_singles_left | 12 | far_service_t |
| 6 | far_singles_right | 13 | near_service_t |

index 14〜19（`net_center`, `left/right_post_base/top`, `center_strap_top`）はアノテーション専用です。

## データ

学習データは `data/court/data_train.json` / `data/court/data_val.json` のアノテーションJSON。`CourtDetectionDataModule` が `config.data.task` で `kp` / `seg` / `line` を切り替えます。

2形式に対応します。

- **Legacy配列形式**: `[{"id": ..., "kps": [[x0, y0], ...]}]`（`kps` を `(14, 2)` として読込）
- **Named keypoints形式**（アノテーションツール出力）: `{"items": [{"image_path": ..., "keypoints": [{"index", "name", "x", "y", "visibility"}, ...]}]}`

Sampleの主キーは `image (3, H, W)`（ImageNet正規化）・`heatmap (K, H, W)`・`keypoints (K, 2)`・`image_id`。data configは `configs/data/court_{kp,seg,line}.yaml`。

## データセット生成（アノテーション）

```bash
# 1. YouTube動画を取得 → AV1/H.264変換 → フレームサンプリング
#    data/court/youtube/annotations/{train,val}.json を named_keypoints 形式で初期化
.venv/bin/python -m src.tasks.court_detection.scripts.prepare_youtube_dataset

# 2. OpenCV UIで手動アノテーション（configs/annotate_youtube_keypoints.yaml）
.venv/bin/python -m src.tasks.court_detection.scripts.annotate_youtube_keypoints
.venv/bin/python -m src.tasks.court_detection.scripts.annotate_youtube_keypoints annotate.split=val
```

`homography_auto_fill: true` のとき、4点以上の接地点アノテーションからホモグラフィで残りの地面点を自動補完します。

## 学習

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.train                          # seg（デフォルト）
.venv/bin/python -m src.tasks.court_detection.scripts.train data=court_kp   loss=kp
.venv/bin/python -m src.tasks.court_detection.scripts.train data=court_line loss=line
.venv/bin/python -m src.tasks.court_detection.scripts.train data=court_kp loss=kp model=court_kp_dino_swin_fpn
```

`configs/training/default.yaml` は `max_epochs=100`、`lr=1e-3`、`bf16-mixed`、AdamW、EarlyStopping（patience=10）。

利用可能なモデルconfig（`configs/model/`）:

| タスク | デフォルト | Swin-L (DINO) | その他 |
|---|---|---|---|
| `kp`（14クラス） | `court_kp` | `court_kp_dino_swin_fpn` / `court_kp_dino_swin_unet` | — |
| `seg`（7クラス） | `court_seg` | `court_seg_dino_swin_fpn` / `court_seg_dino_swin_unet` | `court_seg_dinov3_detr` |
| `line`（1クラス） | `court_line` | `court_line_dino_swin_fpn` / `court_line_dino_swin_unet` | — |

## 可視化

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.visualize                       # kp（デフォルト）
.venv/bin/python -m src.tasks.court_detection.scripts.visualize visualization=seg
.venv/bin/python -m src.tasks.court_detection.scripts.visualize visualization=line \
    visualization.image_source=data/court/images/foo.png
```

ckptと画像ソースは `configs/visualization/{kp,seg,line}.yaml` で指定し、2パネルのGIF（既定 `assets/court_detection/*.gif`、fps=6）を出力します。

## モデル

| モデル | 対応タスク | 実装 |
|---|---|---|
| `CourtHierarchicalModel` | kp / seg / line | `models/hierarchical_model.py`（Encoder×Decoder の組合せ） |
| `DINOv3DETR` | seg 専用 | `models/dinov3_detr.py`（DINOv3 ViT-B/16 + DETRデコーダ、mask-classification） |

`CourtHierarchicalModel` のEncoderは `CourtDefaultEncoder`（軽量4ステージCNN）と `CourtDINOEncoder`（ResNet50/Swin-L等の事前学習backbone）、Decoderは `CourtFPNDecoder` と `CourtUNetDecoder` を組み合わせます。

## 補足

- 実行は `.venv/bin/python -m ...`。Hydra設定は `configs/` から読み込みます。
- DINO/DINOv3 backbone使用時は `third_party/dinov3/` と対応する事前学習チェックポイントが必要です。
