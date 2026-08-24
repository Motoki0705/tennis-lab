# SLCS: Scene Localization in Court System

SLCS は Issue #634 の構造化実動画データセットを読み、単眼の player pose、ball UV、court keypoints と、10フレーム間隔の DINOv3 patch tokens を融合して、コート座標系の player/ball 3D 時系列を同時推定するタスクです。BLCS と PLCS を直列接続せず、frame 内の entity attention と entity ごとの temporal attention を交互に適用します。dataset/clip manifest の正本は `src.tennis_scene.generate_dataset.manifest`、scene schema/archive の正本は `src.tennis_scene.{schema,archive}` です。SLCS 固有の completion marker・必須配列検証だけを `data.annotation` が担当し、foreign schema や error を再exportしません。

## 入出力契約

1 sample は1カメラ・1 temporal window です。`P` は player 数、`T` は window 長、`K` は court keypoint 数、`T_d` は window 内の DINO sample 数、`S` は patch 数です。

| 値 | shape | 表現 |
|---|---|---|
| player pose | `(P,T,17,2)` | normalized image UV |
| ball | `(T,2)` | normalized image UV |
| court | `(T,K,2)` | normalized image UV |
| DINOv3 | `(T_d,S,C)` | `dino_frame_idx` 付き sparse tokens |
| player position | `(P,T,3)` | 選択したcourt normalization contractで正規化 |
| player rotation | `(P,T,2)` | yaw の `(cos,sin)` |
| ball position | `(T,3)` | 選択したcourt normalization contractで正規化 |

すべての観測は visibility/confidence/valid mask を持ちます。DINOv3 tokens は時間方向には補間せず、実 frame index を RoPE position とする cross-attention で伝播します。空間方向は `model.dino_patch_downsample_factor` により、元のDINO特徴空間でbilinear downsampleしてからモデル幅へ次元圧縮できます。factor 2では16×28の448 patchを8×14の112 patchへ圧縮します。player は疑似ラベルの平均 court-Y により near-side、far-side の順へ明示的に並べ替えます。

共有する数式、Hydra選択、単位、metadata互換性、artifact命名、移行手順は
[`src/tasks/base/README.md#court-coordinate-normalization-contract`](../base/README.md#court-coordinate-normalization-contract)
を参照してください。datasetの`SceneResult`と推論で公開するplayer/ball positionは
常にcourt座標のmetre値であり、正規化されるのはmodel境界のpositionだけです。
SLCSのposition uncertaintyはscalar headのまま維持し、`v1`は従来どおり
`mean(scale_xyz)`、isotropicな`v2`は選択済みの共通scaleでmetre換算します。

windowの公開padding契約は`padding_mask (B,T)`、sparse DINO sample軸は`dino_padding_mask (B,T_d)`で、どちらも`True=padding`です。2つの軸は異なるため単一tensorへ統合しません。旧`frame_mask` / `dino_valid`とcaller生成attention maskはadapterでrejectし、entity/time/DINO attention keep-maskはmodel内部でraw padding maskから生成します。評価用`.npz`も`padding_mask`へ破壊的移行し、旧keyへのfallbackは行いません。

モデルとの接続は `model_io` が唯一の境界です。composition 時に sole model と adapter を一度だけ bind し、adapter が必須 key、dtype、rank、全 shape、mask、normalized UV、DINO token spec・frame index semantic を検証してから immutable model call を作ります。model の raw mapping は同じ adapter が `SLCSDecodedOutput` へ decode するため、Lightning・評価・推論 loop は model 名や output key を認識しません。`nn.Module.forward` は検証済み tensor に対する計算だけを行います。

DINO token precompute も同じ境界方針です。`model_io.factory` が backbone と `SLCSFrameTokenIOAdapter` を実行前に一度だけ bind し、adapter が uint8 `(B,H,W,3)` frame を検証・正規化して、`x_norm_patchtokens` を設定済み `(B,S,C)` と照合した後に float16 NumPy array へ変換します。precompute script と clip orchestration は backbone variant、tensor layout、raw output key を扱いません。テストデータの組み立ては production package に置かず、test support から canonical clip export、pseudo annotation/archive、DINO token、dataset index writer を呼びます。

## 学習

```bash
.venv/bin/python -m src.tasks.slcs.scripts.make_splits \
  data.dataset_root=/path/to/dataset data.split_file=/path/to/splits.json

.venv/bin/python -m src.tasks.slcs.scripts.precompute_dino_tokens \
  data.dataset_root=/path/to/dataset

.venv/bin/python -m src.tasks.slcs.scripts.train \
  data.dataset_root=/path/to/dataset data.split_file=/path/to/splits.json

# v2 training command
.venv/bin/python -m src.tasks.slcs.scripts.train \
  court_coordinate_normalization=v2 \
  data.dataset_root=/path/to/versioned/dataset
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

axial trunkの層数は `model.num_shared_layers`、`model.num_position_layers`、`model.num_rotation_layers` で指定します。position branchはplayer/ball位置、rotation branchはplayer yawを担当します。既定の `shared=2, position=0, rotation=0` は従来と同一の全共有構成です。small modelを完全分離する場合は次を指定します。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.train model=small \
  model.num_shared_layers=0 \
  model.num_position_layers=2 \
  model.num_rotation_layers=2
```

## 推論・評価・解析

```bash
.venv/bin/python -m src.tasks.slcs.scripts.predict_clip checkpoint_path=/path/to/model.ckpt
.venv/bin/python -m src.tasks.slcs.scripts.evaluate checkpoint_path=/path/to/model.ckpt
.venv/bin/python -m src.tasks.slcs.scripts.analyze_predictions analysis.arrays=/path/to/eval_arrays.npz
```

評価は player/ball の 3D 誤差、yaw 誤差、速度・加速度・jerk を BLCS/PLCS と比較可能な単位で出力します。解析は誤差分布、時系列誤差、欠損率、uncertainty calibration を保存します。2D overlay は入力観測を描画し、3D prediction の reprojection は calibrated camera が明示された場合だけ行います。

## 検証

```bash
.venv/bin/ruff check src/tasks/slcs tests/unit/tasks/slcs tests/integration/tasks/slcs
.venv/bin/mypy src/tasks/slcs
.venv/bin/python -m pytest tests/unit/tasks/slcs tests/integration/tasks/slcs -q
```

不正 shape、unsupported format、欠損/未完了 annotation、座標範囲違反、DINO spec 不一致、曖昧な player ordering は例外になります。静かな補間・上書き・契約 fallback は行いません。
