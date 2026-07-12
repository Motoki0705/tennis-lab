# SLCS: Sparse-visual Latent Court Scene model

SLCS は Issue #634 の構造化実動画データセットを読み、単眼の player pose、ball UV、court keypoints と、10フレーム間隔の DINOv3 patch tokens を融合して、コート座標系の player/ball 3D 時系列を同時推定するタスクです。BLCS と PLCS を直列接続せず、frame 内の entity attention と entity ごとの temporal attention を交互に適用します。データセット契約は `src.tennis_scene.generate_dataset.manifest` と `pseudo_annotation` が唯一の定義元であり、SLCS はその reader を直接利用します。

## 入出力契約

1 sample は1カメラ・1 temporal window です。`P` は player 数、`T` は window 長、`K` は court keypoint 数、`T_d` は window 内の DINO sample 数、`S` は patch 数です。

| 値 | shape | 表現 |
|---|---|---|
| player pose | `(P,T,17,2)` | normalized image UV |
| ball | `(T,2)` | normalized image UV |
| court | `(T,K,2)` | normalized image UV |
| DINOv3 | `(T_d,S,C)` | `dino_frame_idx` 付き sparse tokens |
| player position | `(P,T,3)` | `COURT_COORD_SCALE_XYZ` で正規化 |
| player rotation | `(P,T,2)` | yaw の `(cos,sin)` |
| ball position | `(T,3)` | `COURT_COORD_SCALE_XYZ` で正規化 |

すべての観測は visibility/confidence/valid mask を持ちます。DINOv3 tokens は補間せず、実 frame index を RoPE position とする cross-attention で時間方向へ伝播します。player は疑似ラベルの平均 court-Y により near-side、far-side の順へ明示的に並べ替えます。

## 学習

```bash
.venv/bin/python -m src.tasks.slcs.scripts.make_splits \
  data.dataset_root=/path/to/dataset data.split_file=/path/to/splits.json

.venv/bin/python -m src.tasks.slcs.scripts.precompute_dino_tokens \
  data.dataset_root=/path/to/dataset

.venv/bin/python -m src.tasks.slcs.scripts.train \
  data.dataset_root=/path/to/dataset data.split_file=/path/to/splits.json
```

split 単位は `recording_id` で、seed と比率を split manifest に保存します。既存 split の上書きには `splits.overwrite=true` が必要です。

1つの小規模データセットを意図的に記憶できるか確認するときだけ、全recordingをtrainへ割り当て、同じwindowをvalidation/testにも使う明示的overfit modeを使用できます。これは汎化性能の評価には使用しません。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.make_splits \
  data.dataset_root=/path/to/dataset data.split_file=/path/to/splits.json \
  splits.overfit=true

.venv/bin/python -m src.tasks.slcs.scripts.train \
  data.dataset_root=/path/to/dataset data.split_file=/path/to/splits.json \
  data.overfit=true
```

損失は confidence-weighted Smooth L1、yaw cosine/wrapped-angle、heteroscedastic Laplace NLL、player/ball jerk、ground penetration を組み合わせます。低品質疑似ラベルは threshold mask と confidence weight で扱います。Issue #634 の契約に calibrated camera がないため、reprojection loss は有効化せず、未校正値も生成しません。

## 推論・評価・解析

```bash
.venv/bin/python -m src.tasks.slcs.scripts.predict_clip checkpoint_path=/path/to/model.ckpt
.venv/bin/python -m src.tasks.slcs.scripts.evaluate checkpoint_path=/path/to/model.ckpt
.venv/bin/python -m src.tasks.slcs.scripts.analyze_predictions predictions_path=/path/to/predictions.npz
```

評価は player/ball の 3D 誤差、yaw 誤差、速度・加速度・jerk を BLCS/PLCS と比較可能な単位で出力します。解析は誤差分布、時系列誤差、欠損率、uncertainty calibration を保存します。2D overlay は入力観測を描画し、3D prediction の reprojection は calibrated camera が明示された場合だけ行います。

## 検証

```bash
.venv/bin/ruff check src/tasks/slcs tests/unit/tasks/slcs tests/integration/tasks/slcs
.venv/bin/mypy src/tasks/slcs
.venv/bin/python -m pytest tests/unit/tasks/slcs tests/integration/tasks/slcs -q
```

不正 shape、unsupported format、欠損/未完了 annotation、座標範囲違反、DINO spec 不一致、曖昧な player ordering は例外になります。静かな補間・上書き・契約 fallback は行いません。
