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

`train_real_rgb_temporal_domain_balanced` は `train_real_rgb_ball_temporal_context` を継承し、train samplingだけを変更します。
`data.domain_sampling.video_domains` でvideo→domainを明示し、品質filter後のtrain window数の逆数を各windowの抽出重みにします。
復元抽出を1epochあたり元のtrain window数だけ行うため、epochのbatch数は維持し、domainは期待値で均衡します（各batch・epochの厳密な50:50ではありません）。
少数domainは重複露出し、多数domainには未抽出windowが生じます。quality値・教師・mask・augmentationは変更しませんが、露出回数に伴う累積gradient寄与は変わります。val/testは従来の逐次走査です。
既定は無効で旧shuffle経路を保持します。旧保存設定でsectionがない場合も明示的に無効へ移行します。
有効時は全train videoのmappingが必須です。train外videoのmappingは許容しますが、設定したdomainに品質filter後のtrain windowが1つもなければ失敗します。空文字・前後空白のdomainは拒否します。
`DomainBalancedSampler` の公開 `domains`・`weights` と `set_epoch(epoch)` で抽出履歴を再計算できます。唯一のseedは `run.seed` で、各epochは `seed + epoch` の独立generatorを使います。
Lightningのepoch通知によりloader再構築・epoch境界resumeでも同じepochの抽出順序を復元します。同じPyTorch実装と同じ順序のtrain metadataが前提で、epoch途中のcursorやaugmentation・worker RNGを含む完全なbit一致resumeは保証しません。
分散学習は未対応でworld size > 1を明示拒否します。このprofileによる精度改善は未検証です。

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

`loss.ball_velocity_weight` を正にすると、ballの予測と教師の一次差分の差を物理速度（m/s）で監督します。`COURT_COORD_SCALE_XYZ` とdatasetの実FPS由来の `timestamp` 差分で換算し、正の `loss.ball_velocity_scale_mps` で割ったXYZ残差にSmooth L1（beta=1、XYZ平均）を適用します。両端の教師がvalid・非paddingで `frame_idx` 差が1のpairだけを、両端confidenceの最小値で加重平均します（分母はconfidence総和）。入力ball visibilityで除外せず、教師の高速運動やbounce自体をゼロに近づけるpriorではありません。有効pairの時間差が非正・非有限の場合や、必要な時間metadataがない場合は明示的に失敗します。

既定のweightは `0.0`（無効）、scaleは単位基準の `1.0` m/sです。weight/scaleの実験値はtrainデータだけで決めます。新設定を含まない旧config/checkpointはこの無効既定で読み込め、従来のlossと推論を維持します。無効時にはvelocity項の計算・logging・時間metadataの要求を追加しません。

`python -m scripts.analysis.calibrate_slcs_ball_velocity --output-root /abs/outputs --training-run slcs/train/<experiment>/<run-id> --output slcs/analyze/<experiment>/<run-id> --device cpu --batch-size 16 --seed 42 --velocity-scale-mps <train-only-scale> --gradient-ratio 0.1` は保存model構成をfresh初期化し、trainから一様非復元抽出した固定batchで、velocityのball位置出力勾配normが既存ball supervised項の指定比率になる重みを `calibration.json` に保存します。保存augmentationとtraining modeを維持し、選択window・RNG手順・元config・両normを記録します。val/testやcheckpointは使わず、overfit・skip-incomplete・既存出力先は拒否します。CUDA実行はtraining queue経由で行います。このCLIは学習設定を変更しません。

比較用の [`train_real_rgb_velocity.yaml`](configs/train_real_rgb_velocity.yaml) はno-ball-smooth・burst24を維持し、[train-only較正](../../../knowledge/nodes/run-slcs-ball-velocity-gradient-calibration-v1.md)のweight/scaleを固定した60epoch profileです。探索中の自動終端testは無効です。係数の決定と、学習後のvalidationによる採否判断は分けて記録します。

[`train_real_rgb_missing_ball_court.yaml`](configs/train_real_rgb_missing_ball_court.yaml) はno-ball-smooth・burst24の60epoch構成で `model.missing_ball_court_context=true` を有効化する単独architecture ablationです。ball欠損の実フレームだけ、既存invisible tokenに、観測court UV（不可視座標は0）とcourt valid flagsのbiasなし線形射影を加算します。court全欠損・paddingでは加算は0です。射影は乱数を消費せずゼロ初期化され、初期のモデル挙動を維持します。既定falseでは追加パラメータはなく旧checkpointをstrict loadできます。自動終端testは無効です。欠損ball tokenにも観測court情報を保持する仮説を検証する設定であり、性能改善は未確認です。

axial trunkの層数は `model.num_shared_layers`、`model.num_position_layers`、`model.num_rotation_layers` で指定します。position branchはplayer/ball位置、rotation branchはplayer yawを担当します。既定の `shared=2, position=0, rotation=0` は従来と同一の全共有構成です。small modelを完全分離する場合は次を指定します。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.train model=small \
  model.num_shared_layers=0 \
  model.num_position_layers=2 \
  model.num_rotation_layers=2
```

[`train_real_rgb_ball_temporal_context.yaml`](configs/train_real_rgb_ball_temporal_context.yaml) は `model.missing_ball_temporal_context=true` の単独入力feature-context仮説を検証する60epoch profileです。no-ball-smooth・burst24を維持し、court residual・velocity loss・自動終端testは無効、resume/initも追加しません。各カメラの連続・等間隔なoffline window内で、欠損実フレームの直前・直後の観測ball tokenをwindow相対indexで線形補間し、ゼロ初期化・biasなし射影を既存invisible tokenに加算します。両側anchorがないedge gap、観測0/1個、padding、rgb_onlyでは追加contextは0です。sourceはcourt+ballの非線形embedding直後で、court residual・entity/time embeddingの前です。両context有効時もsourceは共有せず、court、temporalの順に独立加算します。観測mask・UV・教師は変更せず、他window/cameraの状態、FPS/教師metadata、隠されたball UV、出力平滑化は使用しません。feature補間は物理UV/3D軌道補間とは異なります。既定falseは追加stateを持たず旧checkpointをstrict loadでき、有効時もゼロ初期化は乱数・共有parameter・初期出力を維持します。[TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf) §3.3の時系列欠損修復を参考にした転用仮説であり、同手法の再現や改善の実証ではありません（§4.5は単純な出力線形補間の限界を指摘）。

## 推論・評価・解析

保存済み学習runからvalidation最良checkpointを選んで4入力条件を比較する場合は、
[`scripts.analysis.evaluate_slcs_run`](../../../scripts/analysis/evaluate_slcs_run.py)を使う。
`--help`で明示的な入力・出力rootとsplitの指定を確認できる。既定はvalのみでtestは明示指定とし、
各条件の配列・設定・checkpoint選定記録と実FPSのmotion診断を同じ評価runへ保存する。

同じ学習runをresumeして複数の `last.ckpt` が残る場合は、
`--last-checkpoint logs/version_1/checkpoints/last.ckpt` のように学習run相対で選定元を明示する。
指定したcheckpointのretained validation callbackから最良checkpointを選び、指定fragmentを
`selection.json` に記録する。未指定時は `last.ckpt` が正確に1個の場合だけ評価できる。
更新時刻やversion番号では選ばず、test成績も選定には使わない。
絶対パス、`..`、run外へのsymlink、存在しないfile、`last.ckpt` 以外の指定は拒否する。

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

保存済みの同じ入力条件をモデル間で比較する追加CPU診断は `scripts.analysis.compare_slcs_ball_transitions` を使います。`--baseline` / `--candidate` はそれぞれ `eval_arrays.npz`・`motion.json`・`evaluation_config.yaml` を含む条件ディレクトリ、`--output` は対象評価run内の新規JSONの絶対パスです。教師・mask・confidence・window対応・観測mask・条件・split・物理単位/FPSの一致を検査し、全体とvideo別に4種類のvisibility遷移・教師の高速区間の速度ベクトル誤差と速度biasを保存します。高速区間の閾値 `--fast-speed-mps` はtrain-only統計などから明示し、この評価データにfitしません。window重複は別々に数え、予測速度の低下だけで成功としません。

## PR用可視化

PR用の学習曲線・条件別mean/p95・誤差の経験分布は、完了済み評価をCPUで読み取って生成できます。
評価は5条件（`detector_gap_no_rgb`を含む）が必須です。比較するrun間でwindow・教師・mask・domainを厳密に照合し、数値・入力SHA256・選定receiptを`manifest.json`へ保存します。

```bash
.venv/bin/python -m scripts.analysis.report_slcs_validation \
  --evaluation Baseline=/absolute/outputs/slcs/evaluate/baseline/run-id \
  --evaluation Candidate=/absolute/outputs/slcs/evaluate/candidate/run-id \
  --training Baseline=/absolute/outputs/slcs/train/baseline/run-id \
  --training Candidate=/absolute/outputs/slcs/train/candidate/run-id \
  --output-root /absolute/outputs --output slcs/visualize/comparison/run-id
```

`--training`を省略すると学習曲線だけを省きます。指定時は全labelが必要で、TensorBoardのepochと選定scoreを照合します。
既存出力・欠損・非有限値・対応不一致は拒否し、完了時だけmanifestを保存します。meanは既存SLCSMetrics、p95は同じmasked L2誤差の線形補間percentileです。
重複windowの出現は別々に数え、平滑化・外れ値除外はしません。図は疑似教師との一致度であり、実測3D精度ではありません。個別clipのRGB/3D動画はこのCLIには含みません。

## 検証

```bash
.venv/bin/ruff check src/tasks/slcs tests/unit/tasks/slcs tests/integration/tasks/slcs
.venv/bin/mypy src/tasks/slcs
.venv/bin/python -m pytest tests/unit/tasks/slcs tests/integration/tasks/slcs -q
```

不正 shape、unsupported format、欠損/未完了 annotation、座標範囲違反、DINO spec 不一致、曖昧な player ordering は例外になります。静かな補間・上書き・契約 fallback は行いません。
