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
- `model_io/`: bundle全体の入力、loss、typed prediction契約。KP predictionは `[channel, peak, xy]`、score、validityを明示します。
- `training/`: targetごとのloss/metricを一つのbundleとして集約します。
- `inference/`: single-head predictorはmulti-head checkpointから対象headを明示選択します。
- `visualization/`: bundle-awareなprediction/rendering surface。

設定は `configs/data/default.yaml` をcomposition rootとし、`configs/data/source/` と `configs/data/processing/` を直交してoverrideします。syntheticの`schema=v1|v2|v3`はtyped configで必須で、directory内容から自動推測しません。v2/v3の`train / validation / test`は学習側`train / val / test`へ一意に変換し、空splitやtrajectory group leakageを拒否します。TennisCourtDetectorにtest splitがない既定設定は`data.source.split_mapping.test: null`であり、validationをtestとして代用しません。

## Utilities and scripts

- `src/utils/data/heatmaps.py`: single-peakとall-court multi-peakを共通に扱うdomain-neutral Gaussian heatmap utility。
- `scripts/materialize_targets.py`: source-neutralなseg/line offline materialization。
- `scripts/preview_heatmaps.py`: configured sourceのKP channel/visibilityを使うheatmap preview。
- `scripts/preview_augmentation.py`: 選択target全部を共有geometry上で確認するaugmentation preview。
- `scripts/train.py`: Hydra学習entry point。
- `scripts/visualize.py`: checkpointに保存されたtarget bundleを使うprediction visualization。

YouTube annotation UIは20点を収集しますが、TennisCourtDetector学習契約はordered KP14です。20点annotationからKP14への変換は別の明示的なデータ準備工程を必要とします。
