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

TennisCourtDetector presetは、14点中8点しか一意でなくcourt planeを構成できない`QszoUKyCOHo_600`を`excluded_sample_ids`で明示的にquarantineします。設定したIDがannotation内のちょうど1件に一致しなければsource初期化時に停止するため、データ更新後も古い除外を静かに引き継ぎません。

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
- `model_io/`: bundle全体の入力、loss、typed prediction契約。KP predictionは `[channel, peak, xy]`、score、validityを明示します。
- `training/`: targetごとのloss/metricを一つのbundleとして集約します。
- `inference/`: single-head predictorはmulti-head checkpointから対象headを明示選択します。
- `visualization/`: bundle-awareなprediction/rendering surface。

設定は `configs/data/default.yaml` をcomposition rootとし、`configs/data/source/` と `configs/data/processing/` を直交してoverrideします。syntheticの`schema=v1|v2|v3`はtyped configで必須で、directory内容から自動推測しません。v2/v3の`train / validation / test`は学習側`train / val / test`へ一意に変換し、空splitやtrajectory group leakageを拒否します。TennisCourtDetectorにtest splitがない既定設定は`data.source.split_mapping.test: null`であり、validationをtestとして代用しません。

Model compositionは `model/hierarchical.yaml` をrootとし、encoder、transformer encoder、decoder、dense headを独立したHydra groupとして選択します。既定構成はDINOv3 ViT-B/16、8層のMHA + 2-D RoPE + SwiGLUによるtransformer encoder、DPT decoderです。DPT decoderの出力channelsは512です。

既定のdense headは、DPTのnative-resolution feature上でタスクごとに独立して動くresidual adapterです。各branchは `1x1 projection -> depthwise/pointwise residual block -> 1x1 output` であり、既定はKP / SEG / LINEともhidden channels 256、residual depth 2です。小チャネルのlogitsだけを最後に入力解像度へbilinear補間します。`model/dense_head=linear` は既存の1x1 Conv checkpointを明示的に再構築する場合だけに使用します。

Loss presetは `configs/loss/` で管理し、KP/SEG/LINEのdense項、camera poseのtranslation/rotation/focal項、任意のKP–pose consistency項と各weightを同時に記述します。`default`はdense-only、`pose`はdense lossを維持しながら3種のpose lossを各weight 1.0で有効化します。

pose-only objectiveは専用loss presetを持ちません。`loss=pose`をcomposeし、明示的なoverrideでKP/SEG/LINEのhead weightを0にします。V3 target-court KP14のgeometry・data・head contractは保持されるためdense branchはforwardされますが、dense headにはdense loss由来のgradientは流れません。通常のdense-only設定では0 weightを許可しません。

```bash
# DINOv3 + DPT + LoRA
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data/processing=kp \
  model/encoder=dinov3 model/decoder=dpt training=lora

# DINOv3 patch gridにtransformer refinementを有効化
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data/processing=all \
  model/encoder=dinov3 model/transformer_encoder=default model/decoder=dpt

# KP14 contractを保持するpose-only objective
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court \
  data.source.keypoint_court_scope=target_court \
  data/processing=kp data/augmentation=pose_safe \
  model/encoder=dinov3 model/transformer_encoder=default model/decoder=dpt \
  loss=pose \
  loss.kp.weight=0.0 loss.seg.weight=0.0 loss.line.weight=0.0 \
  loss.consistency.enabled=false
```

Synthetic V3の座標・camera authority・KP semanticの定義は、このconsumer READMEでは再定義しません。正本は上記のSynthetic Court READMEです。

## Utilities and scripts

- `src/utils/data/heatmaps.py`: single-peakとall-court multi-peakを共通に扱うdomain-neutral Gaussian heatmap utility。
- `scripts/materialize_targets.py`: source-neutralなseg/line offline materialization。
- `scripts/preview_heatmaps.py`: configured sourceのKP channel/visibilityを使うheatmap preview。
- `scripts/preview_augmentation.py`: 選択target全部を共有geometry上で確認するaugmentation preview。
- `scripts/train.py`: Hydra学習entry point。
- `scripts/visualize.py`: checkpointに保存されたtarget bundleを使うprediction visualization。

## Target inspection before training

`preview_augmentation.py` はRGB、実際のKP heatmap、7-class SEG、binary LINEを別panelへ描画します。各sampleのJSONにはlossへ渡るtensor shape、可視KP数、Gaussianのpixel sigma / FWHM、SEG class pixel数、LINE foreground比率を保存します。

```bash
# mixed学習のSynthetic側を、augmentation drawも含めて確認
python -m src.tasks.court_detection.scripts.preview_augmentation \
  data/source=synthetic_court \
  data.source.keypoint_court_scope=target_court \
  data/processing=all data/augmentation=pose_safe \
  preview.require_pose=true preview.split=val preview.max_samples=4

# TennisCourtDetector側を確認
python -m src.tasks.court_detection.scripts.preview_augmentation \
  data/source=tennis_court_detector data/processing=all \
  preview.split=train preview.max_samples=4
```

KP Gaussianの `sigma_ratio` は画像対角長に対するsigmaで、学習値は `data.processing.targets` のKP entryが所有します。既定 `0.01` は256x256でsigma約3.62 px、FWHM直径約8.53 pxです。現行LINE schema `court_line_binary_75mm_150mm_v2` は通常線7.5 cm、baseline 15 cmです。旧checkpointが保持する `court_line_binary_v1` は5 cm / 10 cmとして読み取り互換性を維持し、両者のderived target pathは混在しません。

`prepare_youtube_dataset.py` の `workflow.target_preview` は、完成済みYouTube annotationのground KP14からsigmaと物理線幅の候補を比較します。既存annotationだけを読む場合は `enabled=true only=true` を指定します。このYouTube annotation storeは現在のCourt DataModuleへ接続されていないため、このpreviewはtarget候補のauditであり、データを学習へ暗黙に追加しません。

```bash
python -m src.tasks.court_detection.scripts.prepare_youtube_dataset \
  workflow.target_preview.enabled=true \
  workflow.target_preview.only=true \
  workflow.target_preview.sigma_ratios=[0.005,0.01,0.02] \
  workflow.target_preview.line_width_metres=[0.025,0.05,0.075]
```

YouTube annotation UIは20点を収集しますが、TennisCourtDetector学習契約はordered KP14です。20点annotationからKP14への変換は別の明示的なデータ準備工程を必要とします。

## Mixed-source training

`train_mixed`はSynthetic Court V3とTennisCourtDetectorを各train batchへ固定比率で入れます。既定は`synthetic_court=4`、`tennis_court_detector=4`です。KP14は両sourceとも`COURT_KP_NAMES[:14]`へ明示的に正規化され、Synthetic側は1面だけを教師にする`keypoint_court_scope=target_court`を必須とします。source固有schemaの組合せ、semantic channel名、flip permutationのいずれかが変わった場合はmodel構築前に停止します。

```bash
# 両sourceのKP / SEG / LINEだけを学習
python -m src.tasks.court_detection.scripts.train_mixed \
  data/processing=all data/augmentation=pose_safe \
  loss=default \
  run.output_dir=court_detection/mixed-source/dense-only \
  run.test_after_fit=true

# dense lossは全sample、pose lossはSynthetic Court V3 sampleだけで学習
python -m src.tasks.court_detection.scripts.train_mixed \
  data/processing=all data/augmentation=pose_safe \
  loss=pose \
  run.output_dir=court_detection/mixed-source/dense-pose \
  run.test_after_fit=true
```

pose有効時はcollateが必須の`pose_supervision_mask`を生成します。Synthetic Court V3だけが`true`となり、TennisCourtDetector sampleはpose lossとpose metricの双方から除外されます。mask欠落時に全sampleをpose教師として扱うfallbackはありません。TennisCourtDetectorにはtest splitがないため、`test_after_fit`はSynthetic Court V3の明示的test splitだけを評価します。

`run.output_dir`はvariantごとに明示が必須です。config、非queue実行時のtest prediction、その他のrun artifactを異なる学習条件間で上書きしないため、同じ出力先を再利用しないでください。
