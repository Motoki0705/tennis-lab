# Court Detection

テニス映像から `kp / seg / line / semantic_line` を推定します。データsourceとtarget集合は独立に選択し、単一の `CourtDetectionDataset` / `CourtDetectionDataModule` が任意の非空target subsetを処理します。

## Data composition

- `data/source=tennis_court_detector`: yastrebksv/TennisCourtDetector由来の実画像とordered KP14。
- `data/source=synthetic_court`: `schema: v3`を明示したcurrent synthetic source。manifestが明示したRGB配列とlabelsをstrictに読みます。保存codecは下記のSynthetic Court契約に従います。
- `data/source=synthetic_court_v2`: `schema: v2`を明示したlegacy synthetic source。
- `data/source=synthetic_court_v1`: `schema: v1`を明示したcanonical v1回帰source。physical pointを7 semantic multi-peak channelへまとめます。
- `data/processing=kp|seg|line|semantic_line|kp_seg|kp_line|seg_line|all`: 選択したtargetを同じ幾何変換で生成します。`all`が4-head構成です。

Synthetic schema v2/v3では`data.source.court_scope=target_court`を既定とし、sampleの`target_court.binding.court_instance_id`とexact matchする1面だけを4つのdense教師で共有します。`all_courts`を明示するとKP channelとcourt instance inventoryの両方が全accepted courtを保持しますが、現行のsingle-court dense schemaではmaterializationを拒否します。scopeはderived target pathとsource geometry digestにも含まれるため、旧all-court maskをsingle-court教師として再利用しません。target bindingを持たないv1で`target_court`を指定した場合はtyped configuration validationで拒否されます。

source固有のmanifest・annotation・path解決は `data/inputs/`、target固有の構築は `data/processing/targets.py` が所有します。`data/processing/geometry.py` はRGBと全targetに適用する幾何変換をsampleごとに一度だけ決定します。seg/line/semantic_lineはDataset内で生成せず、`data/target_generation/` で事前生成します。

TennisCourtDetector presetは、14点中8点しか一意でなくcourt planeを構成できない`QszoUKyCOHo_600`を`excluded_sample_ids`で明示的にquarantineします。設定したIDがannotation内のちょうど1件に一致しなければsource初期化時に停止するため、データ更新後も古い除外を静かに引き継ぎません。

```bash
# categorical/binaryのdense targetをsource外のderived storeへ生成
python -m src.tasks.court_detection.scripts.materialize_targets \
  data/source=tennis_court_detector data/processing=all

# synthetic sourceの4-head学習
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data/processing=all \
  run.test_after_fit=true

# synthetic v3でcameraの対象コート1面だけを全モダリティの教師にする
python -m src.tasks.court_detection.scripts.train \
  data/source=synthetic_court data.source.court_scope=target_court \
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

既定のdense headは、DPTのnative-resolution feature上でタスクごとに独立して動くresidual adapterです。各branchは `1x1 projection -> depthwise/pointwise residual block -> 1x1 output` であり、既定は4 headともhidden channels 256、residual depth 2です。小チャネルのlogitsだけを最後に入力解像度へbilinear補間します。`model/dense_head=linear` は既存の1x1 Conv checkpointを明示的に再構築する場合だけに使用します。

Loss presetは `configs/loss/` で管理し、KP、court-cell SEG、binary LINE、semantic LINEのdense項、camera poseのtranslation/rotation/focal項、任意のKP–pose consistency項と各weightを同時に記述します。semantic LINEはcategorical CE + multiclass Diceです。`default`はdense-only、`pose`はdense lossを維持しながら3種のpose lossを各weight 1.0で有効化します。

pose-only objectiveは専用loss presetを持ちません。`loss=pose`をcomposeし、明示的なoverrideで4つのhead weightを0にします。V3 target-court KP14のgeometry・data・head contractは保持されるためdense branchはforwardされますが、dense headにはdense loss由来のgradientは流れません。通常のdense-only設定では0 weightを許可しません。

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
  data.source.court_scope=target_court \
  data/processing=kp data/augmentation=pose_safe \
  model/encoder=dinov3 model/transformer_encoder=default model/decoder=dpt \
  loss=pose \
  loss.kp.weight=0.0 loss.seg.weight=0.0 loss.line.weight=0.0 \
  loss.semantic_line.weight=0.0 \
  loss.consistency.enabled=false
```

Synthetic V3の座標・camera authority・KP semanticの定義は、このconsumer READMEでは再定義しません。正本は上記のSynthetic Court READMEです。

## Utilities and scripts

- `src/utils/data/heatmaps.py`: single-peakとall-court multi-peakを共通に扱うdomain-neutral Gaussian heatmap utility。
- `scripts/materialize_targets.py`: source-neutralなseg/line/semantic-line offline materialization。
- `scripts/preview_heatmaps.py`: configured sourceのKP channel/visibilityを使うheatmap preview。
- `scripts/preview_augmentation.py`: 選択target全部を共有geometry上で確認するaugmentation preview。
- `scripts/train.py`: Hydra学習entry point。
- `scripts/visualize.py`: checkpointに保存されたtarget bundleを使うprediction visualization。

## Target inspection before training

`preview_augmentation.py` はRGB、実際のKP heatmap、7-class court-cell SEG、binary LINE、12-class semantic LINEを別panelへ描画します。各sampleのJSONにはlossへ渡るtensor shape、可視KP数、Gaussianのpixel sigma / FWHM、各categorical classの画素数、LINE foreground比率を保存します。

```bash
# mixed学習のSynthetic側を、augmentation drawも含めて確認
python -m src.tasks.court_detection.scripts.preview_augmentation \
  data/source=synthetic_court \
  data.source.court_scope=target_court \
  data/processing=all data/augmentation=pose_safe \
  preview.require_pose=true preview.split=val preview.max_samples=4

# TennisCourtDetector側を確認
python -m src.tasks.court_detection.scripts.preview_augmentation \
  data/source=tennis_court_detector data/processing=all \
  preview.split=train preview.max_samples=4
```

KP Gaussianの `sigma_ratio` は画像対角長に対するsigmaで、学習値は `data.processing.targets` のKP entryが所有します。既定 `0.01` は256x256でsigma約3.62 px、FWHM直径約8.53 pxです。現行single-court LINE schema `court_line_binary_75mm_150mm_single_court_v3` は通常線7.5 cm、baseline 15 cmです。semantic schemaは同じ物理幅を使い、`background / far・near baseline / left・right doubles sideline / left・right singles sideline / far・near service line / center service line / far・near center mark`のcamera-view 12クラスです。交点は生成順で一意に上書きし、水平反転時は左右sideline classだけを交換します。旧all-court schema `court_line_binary_75mm_150mm_v2` と旧5 cm / 10 cm schema `court_line_binary_v1` は別schemaとしてのみ読み取り可能で、現行教師とderived target pathを共有しません。SEGも現行`court_cell_segmentation_single_court_v2`と旧all-court `court_cell_segmentation_v1`を区別します。

KP metricは教師のpoint capacityが1なら各channelの有効画像領域に対してglobal argmaxを1点だけ抽出します。旧all-court形式の`P>1`教師だけがmulti-peak NMSを使用し、この選択はpose lossやLoRAの有無には依存しません。

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

`train_mixed`はSynthetic Court V3とTennisCourtDetectorを各train batchへ固定比率で入れます。既定は`synthetic_court=4`、`tennis_court_detector=4`です。KP14は両sourceとも`COURT_KP_NAMES[:14]`へ明示的に正規化され、Synthetic側は全モダリティで1面だけを教師にする`court_scope=target_court`を必須とします。source固有schemaの組合せ、semantic channel名、flip permutationのいずれかが変わった場合はmodel構築前に停止します。

```bash
# 両sourceの4 dense headだけを学習
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

## Cross-model alignment benchmark

`scripts/benchmark_alignment.py` は、学習済みcheckpointと外部baselineを CPU のみで同一sample集合に推論し、共通metric・可視化・再現artifactを一度に出力します。GPUは受け付けません（`--device`は`cpu`のみ）。

評価対象は2 domainです。`real`は yastrebksv/TennisCourtDetector の held-out **validation** split（公式testではないため、常にvalidationとして表示・報告します）で、`synthetic`は Synthetic Court V3 の明示test split（trajectory groupがtrain/validationとdisjoint）です。sample ID、GT可視性、canonical templateは既存のinput layer（`data/inputs/`）から取得し、benchmark側でannotationやcoverageを再解釈しません。

可視性の意味はdomain間で同一ではありません。`synthetic`の`visible`は`renderer_visible`（occluderを含む描画上の可視性）まで反映したsupervision可視性である一方、`real`はannotation座標が画像内にあるかというin-frame可視性です。したがってcompletenessのようなGT可視点数を分母にする指標の絶対値をdomain横断で直接比較せず、同一domain内のmodel比較として読んでください。

両modelは同じmanifestの同じsampleだけを推論します。manifestは一度だけ書き込み、既存manifest・prediction index・設定fingerprintのいずれかが一致しなければ停止します。予測は1 sampleごとに`predictions/<domain>/<model>/`へ永続化するため、中断後は同じCLIを再実行すれば残りだけを推論します（`--force`で当該cacheを破棄）。

対象modelは`ours`（repo checkpoint）と`tcd`（外部 yastrebksv/TennisCourtDetector）です。外部repoはLICENSEがないため、codeもweightも本repoへcopy・vendor・commitしません。実行時に`--tcd-repo`/`--tcd-checkpoint`で指定し、`tracknet.py`/`postprocess.py`はdynamic importで読み込みます。入力は640x360にresize（BGR, /255）、15 heatmapsの先頭14を使用し、公式validation相当の`low_thresh=155`/`max_radius=30`でHough postprocessします。座標は任意解像度へ`x*W/640`, `y*H/360`で戻し、固定`*2`は使いません。外部repoのhomography postprocessは使用せず、model比較は両者共通のrepo-native RANSAC alignmentで行います。

`ours`側はcheckpointに保存された学習configをそのままreplayします。pose supervisor有効なcheckpointは学習・validationと同じ**long-side** isotropic resize + patch alignmentを再現します（repoの単一画像predictorはshort-side resizeであり、このcheckpointでは一致しません）。checkpointが現行configでreplayできない場合はmodel構築前に停止します。

metricの定義は次の通りです。分母は常に明示します。

- completeness: GT-visible keypointのうちmodelが有効値を出した割合。missingは不正解として扱い、除外しません。
- PCK@d: GT-visible keypointのうち、誤差が画像対角長の`d`以下だった割合（`d = 0.005, 0.01, 0.02, 0.05`）。missingは不正解。
- pair error: 両者が有効なpairのみのpixel誤差と対角長正規化誤差の mean / median / q90。pair数も併記します。
- homography: 4点以上のvalid pairからcanonical court template→image Hを`cv2.findHomography` RANSAC（thresholdは対角長比）で推定し、失敗込みの分母でsuccess rateを報告します。
- homography inliers: `predicted_homography_inliers_on_success`は**成功したfitのみ**を対象にしたinlier数の分布です。失敗したfitは0点として平均に混ぜず、success rateとfailure reasonで数えます。
- line reprojection: GT Hと予測Hで規定コート線を密サンプルし、symmetricなpixel誤差を計算します。frame外へ投影されるtemplate lineはHの外挿になるため、全サンプルの値に加えて「GT投影が画像内に入るline sampleだけ」の値（`line_reprojection_in_frame_*`）と画像内sample比率も併記します。
- doubles IoU: 画像内へclipしたdoubles polygon同士のIoU。

定義できない値（GT可視点が0、GT Hが推定不能、予測Hが失敗など）は0ではなく`null`/`--`とし、`undefined_reasons`に件数を記録します。scene/coverage/visible keypoint数で層別した結果も`metrics.json`に含まれます。

```bash
# 実画像validationと合成testの両方で、repo checkpointと外部baselineを比較
CUDA_VISIBLE_DEVICES='' python -m src.tasks.court_detection.scripts.benchmark_alignment \
  --output-dir /abs/path/to/run \
  --repo-root /abs/path/to/tennis-lab \
  --datasets all --models all \
  --ours-checkpoint outputs/court_detection/.../checkpoints/court-detection-epoch=17.ckpt \
  --tcd-repo /abs/path/to/TCD --tcd-checkpoint /abs/path/to/TCD/model_best.pt

# 決定的samplingで件数を絞ったsmoke
CUDA_VISIBLE_DEVICES='' python -m src.tasks.court_detection.scripts.benchmark_alignment \
  --output-dir /tmp/court-benchmark-smoke --repo-root /abs/path/to/tennis-lab \
  --datasets real_validation --models all --max-samples-per-domain 2 \
  --ours-checkpoint ... --tcd-repo ... --tcd-checkpoint ...
```

`--max-samples-per-domain`は(scene, trajectory group)をround-robinで巡回し、group内のframeはseed依存で回転させるため、件数を絞ってもscene/groupが偏りません。同じseed・同じ入力なら同じmanifestになります。

出力は次の通りです。

- `manifest.json`: 採用sample、selection/quality設定、fingerprint（両modelが共有）
- `provenance.json`: command、torch/cv2/python、device、repo commit、checkpoint/repo hash、model fingerprint、所要時間
- `predictions/<domain>/<model>/`: 1 sample 1 NPZの予測cacheと`index.jsonl`（checksum付き）
- `metrics.json` / `metrics.csv` / `metrics.tex`: 集計値（TeXはpaper側から`\input`できる形。paper配下は変更しません）
- `figures/error_cdf.png`, `figures/summary_bars.png`, `figures/montage_<domain>.png`

montageのsampleはbest/quantile/median/worstとalignment失敗例を固定規則で選び、bestだけを並べません。各行はRGB+GT緑、`ours`、`tcd`、両者のHで投影したコート線を並べます。

`--models`は`ours`/`tcd`単独でも動作し、その場合は選択したmodelをreview primaryとしてfigureを生成します。summary barは実際に評価したmodelだけを描き、未実行のmodelを0本のbarとしては描きません。combined panelのIoU表示は行わず、IoUとH成否は各model panelに明記します。コート線は規定segmentごとに独立したpolylineとして描画するため、segment間を結ぶ線は入りません。

cacheのファイル名はsample IDそのものではなく、`sample-` + percent-encoded IDです（`:`は`%3A`、先頭`-`はそのまま）。sample IDはopaqueな識別子として扱い、`/`・`\`・NULだけを拒否するためpath traversalは表現できず、encodingがinjectiveなので別IDが同じファイルへ衝突しません（旧`:`→`__`置換は`a:b`と`a__b`が衝突し、先頭`-`のYouTube IDを拒否していました）。`index.jsonl`読み込み時にも同名衝突とchecksumを再検査するため、もし将来encoderが非injectiveになっても古いcacheは黙って再利用されず停止します。
