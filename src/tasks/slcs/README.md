# SLCS: Scene Localization in Court System

出力先と実験ごとの設定方針は [タスク出力規約](../OUTPUTS.md) を参照。

Meijiとbroadcastの実RGB学習データを作る手順・品質判定・探索学習profileは
[実RGBデータ生成ガイド](../../tennis_scene/dataset_pipeline/README.md)を参照。

SLCS は Issue #634 の構造化実動画データセットを読み、単眼の player pose、ball UV、court keypoints と、10フレーム間隔の DINOv3 patch tokens を融合して、コート座標系の player/ball 3D 時系列を同時推定するタスクです。BLCS と PLCS を直列接続せず、frame 内の entity attention と entity ごとの temporal attention を交互に適用します。dataset/clip manifest の正本は `src.tennis_scene.generate_dataset.manifest`、scene schema/archive の正本は `src.tennis_scene.{schema,archive}` です。SLCS 固有の completion marker・必須配列検証だけを `data.annotation` が担当し、foreign schema や error を再exportしません。

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

すべての観測は visibility/confidence/valid mask を持ちます。DINOv3 tokens は時間方向には補間せず、実 frame index を RoPE position とする cross-attention で伝播します。空間方向は `model.dino_patch_downsample_factor` により、元のDINO特徴空間でbilinear downsampleしてからモデル幅へ次元圧縮できます。factor 2では16×28の448 patchを8×14の112 patchへ圧縮します。player は疑似ラベルの平均 court-Y により near-side、far-side の順へ明示的に並べ替えます。

windowの公開padding契約は`padding_mask (B,T)`、sparse DINO sample軸は`dino_padding_mask (B,T_d)`で、どちらも`True=padding`です。2つの軸は異なるため単一tensorへ統合しません。旧`frame_mask` / `dino_valid`とcaller生成attention maskはadapterでrejectし、entity/time/DINO attention keep-maskはmodel内部でraw padding maskから生成します。評価用`.npz`も`padding_mask`へ破壊的移行し、旧keyへのfallbackは行いません。

モデルとの接続は `model_io` が唯一の境界です。composition 時に sole model と adapter を一度だけ bind し、adapter が必須 key、dtype、rank、全 shape、mask、normalized UV、DINO token spec・frame index semantic を検証してから immutable model call を作ります。model の raw mapping は同じ adapter が `SLCSDecodedOutput` へ decode するため、Lightning・評価・推論 loop は model 名や output key を認識しません。`nn.Module.forward` は検証済み tensor に対する計算だけを行います。

DINO token precompute も同じ境界方針です。`model_io.factory` が backbone と `SLCSFrameTokenIOAdapter` を実行前に一度だけ bind し、adapter が uint8 `(B,H,W,3)` frame を検証・正規化して、`x_norm_patchtokens` を設定済み `(B,S,C)` と照合した後に float16 NumPy array へ変換します。precompute script と clip orchestration は backbone variant、tensor layout、raw output key を扱いません。テストデータの組み立ては production package に置かず、test support から canonical clip export、pseudo annotation/archive、DINO token、dataset index writer を呼びます。

## 学習

以下はデータ版 `slcs/example_v1` の例です。外部のデータrootを使う場合は
`paths.data_root=/abs/to/data` を追加します。rootと相対パスの契約は
[タスク出力規約](../OUTPUTS.md#入口ごとのroot契約)を参照してください。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.make_splits \
  data.dataset_root=slcs/example_v1 data.split_file=slcs/example_v1/splits.json

.venv/bin/python -m src.tasks.slcs.scripts.precompute_dino_tokens \
  data.dataset_root=slcs/example_v1

.venv/bin/python -m src.tasks.slcs.scripts.train \
  data.dataset_root=slcs/example_v1 data.split_file=slcs/example_v1/splits.json
```

split 単位は `video_id` で、同じマルチカメラ動画から切り出した clip は同一 split に入ります。seed と比率を split manifest に保存します。既存 split の上書きには `splits.overwrite=true` が必要です。

1つの小規模データセットを意図的に記憶できるか確認するときだけ、全videoをtrainへ割り当て、同じwindowをvalidation/testにも使う明示的overfit modeを使用できます。これは汎化性能の評価には使用しません。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.make_splits \
  data.dataset_root=slcs/example_v1 data.split_file=slcs/example_v1/splits.json \
  splits.overfit=true

.venv/bin/python -m src.tasks.slcs.scripts.train \
  data.dataset_root=slcs/example_v1 data.split_file=slcs/example_v1/splits.json \
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

保存済み学習runからvalidation最良checkpointを選んで4入力条件を比較する場合は、
[`scripts.analysis.evaluate_slcs_run`](../../../scripts/analysis/evaluate_slcs_run.py)を使う。
`--help`で明示的な入力・出力rootとsplitの指定を確認できる。既定はvalのみでtestは明示指定とし、
各条件の配列・設定・checkpoint選定記録と実FPSのmotion診断を同じ評価runへ保存する。

`--gap-no-rgb` を指定すると、第5条件 `detector_gap_no_rgb` を追加する。
既存 `detector_gap` と同じ有効窓中央1/3のball・全player観測欠損に `no_rgb` を重ね、court観測は維持する。
通常の4条件比較に加え、各splitの `gap_rgb_comparison/comparison.json` と `comparison.csv` に
`detector_gap` 対 `detector_gap_no_rgb` の位置・yaw誤差を全体・domain・video別で保存する。
checkpoint SHA256、教師・mask・weight・window対応の完全一致を検証し、
`detector_gap_minus_condition_*` が負ならRGBありのgap条件を支持する。
第5条件も配列・設定・motion診断を保存し、`--ball-train-mean` 併用時は定数baselineとも比較する。
この診断は疑似教師との一致度であり、実測3D精度や因果効果ではない。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.predict_clip \
  predict.checkpoint=slcs/example.ckpt \
  data.dataset_root=slcs/example_v1 \
  predict.clip_id=video_000/clip_000 predict.camera_id=cam0

.venv/bin/python -m src.tasks.slcs.scripts.evaluate \
  evaluate.checkpoint=slcs/example.ckpt \
  data.dataset_root=slcs/example_v1 data.split_file=slcs/example_v1/splits.json \
  evaluate.output_dir=slcs/evaluate/example/s42-001

.venv/bin/python -m src.tasks.slcs.scripts.analyze_predictions \
  analysis.arrays=slcs/evaluate/example/s42-001/eval_arrays.npz
```

標準評価は player/ball の 3D 位置誤差と yaw 誤差を出力します。追加のCPU診断は `evaluation.motion.summarize_motion(arrays, fps_by_clip, position_representation="normalized_court")` を使用します。`fps_by_clip` は評価対象と完全一致する `(video_id, full clip_id) -> FPS` の明示mappingです。有効な連続フレームの組だけから速度・加速度・jerk（m/s、m/s²、m/s³）の予測値・教師値・誤差を計算し、位置のXYZ標準偏差と標準偏差ノルム比も全体・video・camera別に返します。重複windowは別々に数え、playerは追跡IDではなくnear/far slot別です。分散比は成功指標ではなく、教師分散が0の場合は未定義です。解析は誤差分布、時系列誤差、欠損率、uncertainty calibration を保存します。2D overlay は入力観測を描画し、3D prediction の reprojection は calibrated camera が明示された場合だけ行います。

`scripts.analysis.evaluate_slcs_run --ball-train-mean` は保存済み学習設定のtrain splitだけからconfidence-weighted arithmetic meanのball定数を推定し、`ball_train_mean_fit.json` と各split/input conditionの `ball_train_mean_comparison.json` を保存します。fitはproductionのwindow/quality設定を保持し、augmentationなしで `(video, clip, camera, frame)` の重複教師・mask・weightの一致を確認して集約します（cameraは別観測）。比較は既存headlineと同じ非加重valid window occurrencesで、overall/domain/video別のEuclidean誤差をm単位で出力します。この平均は二乗距離を最小化する定数で、平均Euclidean距離の最適定数ではありません。train評価はin-sample、overfit設定と不完全annotationのskip設定は拒否します。教師を読むCPU処理だけは `require_dino=False` とし、DINO cacheを読まず検証もしません。モデルの各入力条件と通常のDINO検証は変更せず、評価教師・window metadataをproduction splitと照合します。

## 検証

```bash
.venv/bin/ruff check src/tasks/slcs tests/unit/tasks/slcs tests/integration/tasks/slcs
.venv/bin/mypy src/tasks/slcs
.venv/bin/python -m pytest tests/unit/tasks/slcs tests/integration/tasks/slcs -q
```

不正 shape、unsupported format、欠損/未完了 annotation、座標範囲違反、DINO spec 不一致、曖昧な player ordering は例外になります。静かな補間・上書き・契約 fallback は行いません。
