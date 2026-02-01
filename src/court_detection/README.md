# Court Detection (CourtKP20)

`src/court_detection` は、テニス映像からコートのキーポイント（20点）を検出するタスク実装です。

## 目的 / 想定入出力

- **入力**: テニスコート画像（RGB）
- **出力**: 20個のコートキーポイントの2D座標 + 可視性フラグ

学習データは `src.tools.annotate_court_keypoints` で生成した JSON アノテーションを前提とします。

### Predictor 出力形式

**Predictor 返り値（`CourtKeypointPredictor.predict()`）:**

推論結果は `dict[str, torch.Tensor]` 形式で返されます。全てのテンソルは CPU 上にあります。

| キー | 形状 | 型 | 説明 |
|------|------|-----|------|
| `keypoints` | `(K, 2)` | `torch.Tensor` | キーポイント座標（ピクセル空間）、K=20 |
| `visibility` | `(K,)` | `torch.Tensor` | 可視性確率（0-1の範囲） |
| `heatmaps` | `(K, H, W)` | `torch.Tensor` | ヒートマップ（`return_heatmaps=True` の場合のみ） |

**注意**: 
- すべてのテンソルは CPU に配置されます（統合側での device 変換は不要）
- キーポイント座標は元画像のサイズにスケーリング済みです

### キーポイント仕様 (CourtKP20)

`src/utils/geometry/court.court_keypoints_3d()` で定義される20点：

| Index | Name | Description |
|-------|------|-------------|
| 0 | far_doubles_corner_left | 奥側ダブルスコーナー（左） |
| 1 | far_doubles_corner_right | 奥側ダブルスコーナー（右） |
| 2 | near_doubles_corner_left | 手前ダブルスコーナー（左） |
| 3 | near_doubles_corner_right | 手前ダブルスコーナー（右） |
| 4 | far_singles_corner_left | 奥側シングルスコーナー（左） |
| 5 | near_singles_corner_left | 手前シングルスコーナー（左） |
| 6 | far_singles_corner_right | 奥側シングルスコーナー（右） |
| 7 | near_singles_corner_right | 手前シングルスコーナー（右） |
| 8 | far_service_left | 奥側サービスライン端点（左） |
| 9 | far_service_right | 奥側サービスライン端点（右） |
| 10 | near_service_left | 手前サービスライン端点（左） |
| 11 | near_service_right | 手前サービスライン端点（右） |
| 12 | far_service_T | 奥側サービスT |
| 13 | near_service_T | 手前サービスT |
| 14 | net_center | ネット中央（地面） |
| 15 | net_post_left_base | 左ネットポスト（下） |
| 16 | net_post_left_top | 左ネットポスト（上） |
| 17 | net_post_right_base | 右ネットポスト（下） |
| 18 | net_post_right_top | 右ネットポスト（上） |
| 19 | center_strap_top | センターストラップ（上） |

**注**: このキーポイント仕様は `src/utils/geometry/constants.py` の `COURT_KP_NAMES` / `COURT_KP_IDX` として定義されており、プロジェクト全体で参照すべき canonical reference となっています。各モジュール（CourtDetection / PLCS / BLCS / Rendering など）はこの定義を参照することで、indexの解釈が統一されます。

## ディレクトリ構成

```
src/court_detection/
├── configs/                          # Hydra 設定ファイル群
│   ├── train.yaml                    # 学習メイン設定
│   ├── visualize.yaml                # 可視化設定
│   ├── run/                          # 実行時設定
│   │   └── default.yaml
│   ├── model/                        # モデル設定
│   │   └── vit_heatmap.yaml          # ViT encoder/decoder + Heatmap
│   ├── data/                         # DataModule 設定
│   │   └── default.yaml
│   ├── training/                     # 学習ハイパーパラメータ
│   │   └── default.yaml
│   ├── loss/                         # 損失関数設定
│   │   └── default.yaml
│   └── visualization/
│       └── default.yaml
│
├── scripts/                          # 実行スクリプト
│   ├── train.py                      # モデル学習
│   └── visualize.py                  # 推論結果の可視化
│
├── models/                           # モデル実装
│   ├── court_keypoint_model.py       # ViT encoder/decoder モデル
│   └── components/
│       ├── backbones.py              # 旧バックボーン実装
│       └── heads.py                  # 旧ヘッド実装
│
├── data/                             # データセット・DataModule
│   ├── dataset.py                    # Dataset
│   └── datamodule.py                 # LightningDataModule
│
├── training/                         # 学習関連
│   ├── lightning_module.py           # LightningModule
│   ├── losses.py                     # 損失関数
│   └── metrics.py                    # 評価指標
│
├── inference/                        # 推論
│   ├── predictor.py                  # 推論クラス
│   └── visualization.py              # 可視化ヘルパー
│
```

## 主要コンポーネントの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ train.py                                                        │
│   ├── data/datamodule.py           (DataModule)                 │
│   ├── models/court_keypoint_model.py (モデル)                   │
│   └── → outputs/court_detection/*/checkpoints/                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ visualize.py / inference/predictor.py                           │
│   └── inference/visualization.py   (描画)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 実行コマンド

### 学習

```bash
uv run python -m src.court_detection.scripts.train
```

### 可視化

```bash
uv run python -m src.court_detection.scripts.visualize
```

## 手動アノテーション

実映像からアノテーションを作成する場合は `src.tools.annotate_court_keypoints` を使用：

```bash
uv run python -m src.tools.annotate_court_keypoints \
    input_path=data/raw/court_image.jpg \
    output.output_dir=data/court_keypoints
```

## 外部提供 API

学習済みモデルを用いた推論 API は `src/court_detection/inference/predictor.py` を参照：

```python
from src.court_detection.inference.predictor import CourtKeypointPredictor

predictor = CourtKeypointPredictor.from_checkpoint("path/to/checkpoint.ckpt")
keypoints, visibility = predictor.predict(image)
```
