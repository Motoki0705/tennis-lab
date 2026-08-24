# Court Detection

テニス映像から `kp / seg / line` を推定します。データsourceとtarget集合は独立に選択し、単一の `CourtDetectionDataset` / `CourtDetectionDataModule` が任意の非空target subsetを処理します。

## Data composition

- `data/source=tennis_court_detector`: yastrebksv/TennisCourtDetector由来の実画像とordered KP14。
- `data/source=synthetic_court`: `schema: v3`を明示したcurrent synthetic source。manifestが公開した`rgb.npy`とlabelsだけをstrictに読みます。
- `data/source=synthetic_court_v2`: `schema: v2`を明示したlegacy synthetic source。
- `data/source=synthetic_court_v1`: `schema: v1`を明示したcanonical v1回帰source。physical pointを7 semantic multi-peak channelへまとめます。
- `data/processing=kp|seg|line|kp_seg|kp_line|seg_line|all`: 選択したtargetを同じ幾何変換で生成します。

Synthetic schema v2/v3では`data.source.keypoint_court_scope=all_courts|target_court`でKP教師に含めるコートを選択できます。既定の`all_courts`は全accepted courtを14 semantic channelのpoint軸へ保持します。`target_court`はsampleの`target_court.binding.court_instance_id`とexact matchする1面だけをpoint軸へ保持します。このoptionはKP教師だけに作用し、全コートの`court_instances`と事前生成するseg / lineの参照・内容には作用しません。v1で`target_court`を指定した場合はtyped configuration validationで拒否されます。

source固有のmanifest・annotation・path解決は `data/inputs/`、target固有の構築は `data/processing/targets.py` が所有します。`data/processing/geometry.py` はRGB、KP、seg、lineに適用する幾何変換をsampleごとに一度だけ決定します。seg/lineはDataset内で生成せず、`data/target_generation/` で事前生成します。

```bash
# 両dense targetをsource外のderived storeへ生成
python -m src.tasks.court_detection.scripts.materialize_targets \
  data/source=tennis_court_detector data/processing=seg_line

# synthetic sourceの3-head学習
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data/processing=all \
  run.test_after_fit=true

# synthetic v3でcameraの対象コート1面だけをKP教師にする
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data.source.keypoint_court_scope=target_court \
  data/processing=kp

# legacy synthetic v2のdense targetを学習前にsource外へ事前生成
python -m src.tasks.court_detection.scripts.materialize_targets \
  data/source=synthetic_court_v2 data/processing=seg_line

# KP-only DINOv3 + DPT + LoRA
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data/processing=kp \
  model/encoder=dinov3 model/decoder=dpt training=lora
```

`synthetic_court`の`dataset.json`やsample fileはmaterializationで変更しません。生成物はsource root外の `data.processing.derived_target_root` 以下へ、source kind・sample key・target schemaを含む安定pathで保存します。Dataset/DataModuleはmaskを生成せず、requested dense targetのPNG・provenance metadata・digestが欠落またはstaleならDataLoader worker起動前に停止します。

Synthetic schema v1/v2/v3の生成・publication・semantic contractの正本は [`src/synthetic_data_generation/dataset/court/README.md`](../../synthetic_data_generation/dataset/court/README.md) です。このREADMEではconsumer設定、事前生成、学習手順だけを管理します。

## Model and runtime

- `models/hierarchical_model.py`: shared encoder/decoder trunkと、`CourtTargetBundleSpec`から導出したhead群。
- `models/query_encoder/`: 明示的なablation variant。DINOv3 patch boundary、task encoder、`linear / progressive / dpt` decoder、dense headとpose10D headを独立したHydra groupとしてcomposeします。production defaultは変更しません。
- `model_io/`: bundle全体の入力、loss、typed prediction契約。KP predictionは `[channel, peak, xy]`、score、validityを明示します。
- `training/`: targetごとのloss/metricを一つのbundleとして集約します。
- `inference/`: single-head predictorはmulti-head checkpointから対象headを明示選択します。
- `visualization/`: bundle-awareなprediction/rendering surface。

設定は `configs/data/default.yaml` をcomposition rootとし、`configs/data/source/` と `configs/data/processing/` を直交してoverrideします。syntheticの`schema=v1|v2|v3`はtyped configで必須で、directory内容から自動推測しません。v2/v3の`train / validation / test`は学習側`train / val / test`へ一意に変換し、空splitやtrajectory group leakageを拒否します。TennisCourtDetectorにtest splitがない既定設定は`data.source.split_mapping.test: null`であり、validationをtestとして代用しません。

Query variantは明示的に選択します。poseありの初期比較はV3 target-court KP、`pose_safe` augmentation、query lossを同時にcomposeする必要があります。model componentをraw overrideする場合はpresetとの不一致を隠さないよう`model.preset=raw`も明示します。

```bash
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court \
  data.source.keypoint_court_scope=target_court \
  data/processing=kp data/augmentation=pose_safe \
  model=query_encoder_base loss=query_pose

# componentを独立overrideする例
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court \
  data.source.keypoint_court_scope=target_court \
  data/processing=kp data/augmentation=pose_safe \
  model=query_encoder_base model.preset=raw \
  model/task_encoder=query_base model/decoder=query_progressive_small \
  model.decoder.tap_indices=[7] model/heads=query_base loss=query_pose
```

Synthetic V3の座標・camera authority・KP semanticの定義は、このconsumer READMEでは再定義しません。正本は上記のSynthetic Court READMEです。

## Query ablation workflow

3つのentry pointはいずれもHydra configだけを使用し、GPU学習を直接起動しません。

```bash
# 1. 3 phase × 3 seedのordered manifestを生成
python -m src.tasks.court_detection.scripts.run_query_ablation

# encoder結果から選んだdepthを解決し、decoder phaseをqueue-readyにする
python -m src.tasks.court_detection.scripts.run_query_ablation \
  ablation.selected.encoder_depth=4

# decoder結果も選択した完全解決manifest
python -m src.tasks.court_detection.scripts.run_query_ablation \
  ablation.selected.encoder_depth=4 \
  ablation.selected.decoder_family=linear \
  ablation.selected.decoder_size=base

# 2. 同一256x256入力contractで1 candidateをprofile
python -m src.tasks.court_detection.scripts.profile_query_model

# 3. 全run・全metric・GPU profileと明示的な採用判断を検証して集約
python -m src.tasks.court_detection.scripts.summarize_query_ablation \
  summary.adoption.supervision=kp+pose \
  summary.adoption.rationale='complete three-seed evidence supports joint pose supervision'
```

manifestの順序はencoder depth `1/2/4/8`、選定encoderを固定したdecoder `linear/progressive/dpt × tiny/small/base`、選定architectureを固定したsupervision `kp / kp+pose / all / all+pose`です。各runは15 epoch、seed `42/43/44`、256×256 isotropic letterbox、`preserve_fx_fy=true`、hflip/affine/shear/perspectiveなしを固定します。未選定の後続phaseはplaceholderと`queue_ready=false`を保持し、暗黙の構成には置換しません。

manifest内の`command_argv`は、repository root共有のtraining queueへ順番にenqueueするための入力です。`run_query_ablation.py`自身はqueue workerもtraining processも開始しません。profileはdecoder/trainable/total parameters、明示定義のdecoder MACs、decoder/end-to-end latency、peak memoryを記録します。CPU profileはdiagnosticと明記され、採用判断のGPU latency evidenceとしてsummarizerに受理されません。summaryはscaling/Pareto tableとplot、および全candidateの`adopted / non_adoption_reason`を出力し、run・seed・metric・phaseが1件でも欠ければ停止します。

## Query KP–pose consistency scaling workflow

Issue #790のscaling/formal comparisonは、上記#779の51-run routeを変更しない独立manifestです。Phase 1はencoder depth `1/2/4/8 × seed 42/43/44`、Phase 2は選定encoderを固定したdecoder `linear/progressive/dpt × tiny/small/base × 3 seed`、Phase 3は選定architectureを固定した`direct-all / joint-both / joint-stopgrad-pose / joint-stopgrad-dense × 3 seed`です。全phaseでV3 target-court singleton KP14、`KP/LINE/SEG + translation/rotation/focal`直接教師、pose-safe 256×256、15 epochを固定します。

```bash
# Phase 1だけがqueue-readyなmanifest
python -m src.tasks.court_detection.scripts.run_query_consistency_ablation

# 完了したPhase 1 evidenceからencoder選定を検証・集約
python -m src.tasks.court_detection.scripts.summarize_query_consistency_ablation \
  summary.phase=encoder_scaling

# 選定encoderを明示してPhase 2を解決
python -m src.tasks.court_detection.scripts.run_query_consistency_ablation \
  consistency_ablation.selected.encoder_depth=4

# Phase 2完了後にdecoder選定を検証・集約
python -m src.tasks.court_detection.scripts.summarize_query_consistency_ablation \
  summary.phase=decoder_scaling

# 選定encoder/decoderをすべて明示してPhase 3を解決
python -m src.tasks.court_detection.scripts.run_query_consistency_ablation \
  consistency_ablation.selected.encoder_depth=4 \
  consistency_ablation.selected.decoder_family=linear \
  consistency_ablation.selected.decoder_size=tiny

# 全51 runの採用/不採用条件を自動判定
python -m src.tasks.court_detection.scripts.summarize_query_consistency_ablation \
  summary.phase=consistency_ablation
```

各ready runはtraining用`command_argv`とcapacity用`profile_command_argv`を公開します。どちらもrepository root共有training queueへそのまま渡せるargvであり、manifest generator自身はGPU processやqueue workerを開始しません。`command_argv`は`run.output_dir`、run固有artifact root、`run.test_after_fit=true`を明示します。未選定phaseはargvを持たず`queue_ready=false`となり、placeholderや既定architectureへ置換されません。

Frozen ruleがencoder depth 1を選んだ場合、DPTに必要な2個以上のunique tapを作れません。この場合はdepthを暗黙に変更せず、Phase 2 manifestの解決を明示エラーにしてsummaryへblockerを記録します。

summarizerはtest `metrics.json`のcanonical metric、TensorBoardのbranch gradient/train-step/CUDA-memory scalarとloss curve、GPU capacity profileを別々の必須sourceとして収集します。legacy `line_iou`は`line_dice`のaliasとして受理しません。1 run/seed/metric/diagnostic/profileでも欠けると停止し、all-run/scaling/Pareto tableとplot、Phase 3ではIssueのGT改善・直接metric劣化・dense metric・cost上限を使った採用判断を出力します。

## Utilities and scripts

- `src/utils/data/heatmaps.py`: single-peakとall-court multi-peakを共通に扱うdomain-neutral Gaussian heatmap utility。
- `scripts/materialize_targets.py`: source-neutralなseg/line offline materialization。
- `scripts/preview_heatmaps.py`: configured sourceのKP channel/visibilityを使うheatmap preview。
- `scripts/preview_augmentation.py`: 選択target全部を共有geometry上で確認するaugmentation preview。
- `scripts/train.py`: Hydra学習entry point。
- `scripts/visualize.py`: checkpointに保存されたtarget bundleを使うprediction visualization。
- `scripts/profile_query_model.py`: 固定input contractのquery capacity/runtime profiler。
- `scripts/run_query_ablation.py`: queueへ渡すordered argv manifest generator。
- `scripts/summarize_query_ablation.py`: 完了runのscaling/Pareto summary generator。
- `scripts/run_query_consistency_ablation.py`: #790の段階解決manifest generator。
- `scripts/summarize_query_consistency_ablation.py`: #790 evidence collector/scaling/adoption summary。

YouTube annotation UIは20点を収集しますが、TennisCourtDetector学習契約はordered KP14です。20点annotationからKP14への変換は別の明示的なデータ準備工程を必要とします。
