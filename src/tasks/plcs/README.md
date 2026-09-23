# PLCS

出力先と実験ごとの設定方針は [タスク出力規約](../OUTPUTS.md) を参照。

三角測量の初期COCO17を補正する残差モデルは、本READMEの「Triangulation residual」を参照。実装は既存のdata・geometry・models・model_io・training・inference・visualizationレイヤーに配置します。

2D の人物 pose とコート keypoint から、コート座標系でのプレイヤー `position`/`rotation`（および任意で canonical 3D pose）を推定するタスクです。ACCAD (AMASS/SMPL-H) または GVHMR モーションと仮想カメラから学習データを合成する generator、frame/sequence/multiview の各モデル、Lightning 学習、推論、可視化までを一貫して提供します。

## Court keypoint contract

CourtKP20 の version、camera-local disk ordering、camera-local model inputs、
metadata と checkpoint の exact-match 規約は共有正本
[`src/tasks/base/generate_dataset/README.md`](../base/generate_dataset/README.md)
を参照してください。PLCS 固有の差分は、disk の `court_kp_uv` / `court_kp_vis`
がそれぞれ `(T,20,2)` / `(T,20)`、standard sample がcamera-localの20点、tracking
sample がcamera-localの先頭14点を使うことです。reference transform は position、
heading、court-space world joints に適用し、player-local `canonical_pose_3d` と
human UV/visibility には適用しません。

## Motion source contract

dataset generator へ渡すモーションの唯一の境界は
`motion/contracts.py` の `Coco17MotionClip` です。全source adapterは、右手系・metre・
Z-upのCOCO-17 joint、native timestamp/FPS、root translation、完全なroot rotation、
joint confidence、frame validityへ変換します。COCO-17にはpelvisがないため、rootの
並進・回転を17点から再推定しません。ACCAD adapterはSMPL-Hのroot信号を、GVHMR
adapterはglobal SMPL-Xの`transl`/`global_orient`をそれぞれ保持します。

`rotation=(cos θ, sin θ)`は、`motion/geometry.py`の`root_yaw()`が求める
root-local `+X`軸の方位を符号化したyawです。ACCAD/GVHMRのSMPL身体前方は
local `+Z`で、Z-upかつyawを除いたcanonical poseでは基準前方が`-Y`になります。
可視化の矢印はこの基準前方を回転した`(sin θ, -cos θ)`を使います。
保存されたrotation pair自体をXY方向として描くと90度横を向きます。

保存形式はpickleを使わないversioned `*.motion.npz` です。generatorの
`motion_sources` entryは`format`、`paths`、`weight`を明示し、現在は
`amass_smplh_v1`と`coco17_motion_v1`を登録しています。scene生成側はsource固有の
SMPL形式を扱いません。ACCAD専用profileは`motion_sources=accad`、混合profileは
`motion_sources=accad_gvhmr`です。曖昧な`motion_sources/default.yaml`は廃止しました。

scene artifactはnative FPSを保持します。学習Datasetだけが
`augmentation.frame_rate`の候補（既定は28/30/60 Hz）をsampleし、2D観測、3D
target、visibility、instance lifecycleを同じ時刻へ同期resampleします。評価時は
resampleしません。これにより入力ファイルを60 fpsへ固定せず、時間スケールへの
耐性を学習時に付与します。1本の共有timelineへ複数人を配置するsceneでは、全員を
同じnative FPSのsource群から選び、異なるrateをframe indexだけで混ぜることを禁止
します。ACCAD/GVHMRの混合比はscene間で保たれます。

GVHMR抽出の入口は`configs/extract_gvhmr_motions.yaml`です。`dataset`の選択は必須で、
`configs/dataset/<name>.yaml`が入力rootと選手選択設定を指定します。meiji_3camでは
`configs/gvhmr_motion/meiji_3cam.yaml`を選手選択の正本とします。
人物検出はDINO 4-scale Swin-L、対応付けはBoT-SORTを使います。cam0は
使わず、cam1とcam2で各カメラの手前選手をfootpoint ROIで1人ずつ選びます。
ROI内の単一選手についてtrack IDの分断を明示的に連結し、各入力clipから2本の
motion、raw GVHMR sidecar、品質recordを生成します。抽出pipeline versionは成果物と
collection manifestの双方へ保存し、異なるdetectorによる再開・混在を拒否します。
容量を制御するため、dense mesh vertex列とrendered videoは保存しません。raw sidecarも
再現に必要なSMPL parameter・camera intrinsic・2D観測だけに限定します。
各clipの推論前と保存前に、1 GiBの予備容量とclipの保守的な保存容量を確認し、
不足時は完了済み成果物を保持して停止します。容量確保後、同じcommandで再開できます。

```bash
# Local GPUでは共有training queue経由でこのcommandを実行する。
.venv/bin/python -m src.tasks.plcs.scripts.extract_gvhmr_motions \
  dataset=meiji_3cam run.seed=42

# 推論せず合成済み設定を確認する。
.venv/bin/python -m src.tasks.plcs.scripts.extract_gvhmr_motions \
  dataset=meiji_3cam --cfg job --resolve
```

別データセットは`configs/dataset/<name>.yaml`と選手ROI設定を追加して`dataset=<name>`
で切り替えます。入力は共通のdataset/clip manifest形式が必要です。モデル既定値は
`models.config`の共通設定から読み、`models.runtime_overrides`で設定ファイル上の
差分を指定できます。独自argparseは使いません。Pythonでは同じDictConfigを
`scripts.extract_gvhmr_motions.run_extraction()`へ渡せます。

出力rootに`config.yaml`（絶対rootを保存したHydra再実行用設定）と
`reproducibility.json`（実効モデル設定、重み・人体モデル・実装のSHA256、依存版、
seed、Git commitと未コミットのコード差分）を保存します。各motionには動画とclip manifestのSHA256も記録し、
再開時に照合します。seedはsource IDから決定するため処理順・subsetに依存しません。
設定や重み・実装が変わった場合は新しい`run.output_dir`が必要です。旧抽出結果には
この検証情報がないため混在を拒否し、既定の出力先も`<dataset>_repro_v1`へ分けています。
乱数と決定的演算を固定しますが、異なるGPU・依存環境間のbit単位一致は保証しません。

保存設定での再実行は`--config-dir /absolute/output/root --config-name config`で行います。
worktreeで共有モデルを使う場合は`paths.data_root`、`paths.checkpoint_root`、
`paths.external_asset_root`を共有rootの絶対パスへ設定し、`paths.project_root`は実装を
読み込むworktreeに保ちます。
抽出先を変更した場合は、データ生成時の`motion_sources.tennis.paths`にもその
data-root相対パスを指定してください。既存の混合profileは既に生成済みの
`plcs/motions/gvhmr/meiji_3cam`を参照しています。

## Modules

### configuration
- **`configuration.py`**: training・analysis・visualization runtime boundary。共有 contract を消費し、generation package は import しない。
- **`configuration_contracts.py`**: training と standalone generation が共有する path roots と generation component の型付き契約。両 runtime 設定より下位に置き、相互 import を作らない。

### data/
- **`dataset.py`**: `SceneDataset`。sceneをcamera-time基準のcanonical sample(`human_kp`/`court_kp`/`position`/`rotation`等)に変換。augmentation前の`human_kp_target`/`human_vis_target`と選択camera parameterも保持し、2D reprojection supervisionへ渡す。
- **`datamodule.py`**: `PLCSDataModule`。model非依存のcanonical `(B,V,T,...)` batchを構築し、profile固有変換は行わない。
- **`frame_rate_augmentation.py`**: native timingを保持したsceneから学習時のtarget FPSをsampleし、continuous/discrete/heading信号を意味に応じて同期resampleする。
- **`augmentation/observation.py`**: `PLCSObservationAugmentation`。UVノイズ・時間jitter・可視性dropout等8段のパイプライン。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`targets.py`**: `build_coco17_world_targets()`。canonical poseまたはAthletePose3DからCOCO17ワールド座標targetを構築。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、pose観測をnoise/dropout/false-positiveで破損してからcamera-local trackingにより固定幅`Q`へ変換するDataset/DataModule。target lifecycle packingは観測associationと独立です。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成します。val/testは常に`scene_dir`上の固定splitを使います。
- **`tracking_augmentation.py`**: 固定`Q`へ変換する前の物理幅pose detectionへcorruptionを適用し、false-positive provenanceを`-1`にするadapter。synthetic-only carrierの容量制御は[共有tracking contract](../base/README.md)に委譲する。
- **`types.py`**: `PLCSBatch`/`PLCSSceneMeta` のバッチ・meta契約。

### models/
- **各model module**: 実装classのcanonical import先。package rootは内部classや旧factoryをre-exportしない。
- **`plcs_model.py`**: `PLCSModel`。単視点frame向けdecoder-only Transformer(court+playerトークン)。
- **`plcs_multiview_axial_reference_model.py`**: `PLCSMultiViewAxialReferenceModel`。reference selectorを第3 RoPE軸に持ち、指定cameraの特徴からposition・rotation・canonical poseを読む。
- **`plcs_multiview_axial_model.py`**: `PLCSMultiViewAxialModel`。camera軸/time軸交互self-attention(共有readout)。
- **`plcs_multiview_axial_foot_residual_model.py`**: split trunkに可視性・身体相対座標・足首ground priorの埋め込みを加え、位置をprior + XYZ残差として出力。観測だけから求める幾何は `geometry/footpoint.py` が所有する。
- **`plcs_multiview_axial_split_model.py`**: `PLCSMultiViewAxialSplitModel`(issue #518)。rotation/pose trunkを分離。
- **`plcs_multiview_axial_camtoken_model.py`**: `PLCSMultiViewAxialCamTokenModel`(issue #576)。head別に別camera tokenを読む。
- **`plcs_track_query_model.py`**: `PLCSTrackQueryModel`。object streamをviewごとに1 tokenへ圧縮し、FFN-free attention block、`Q+V` spatial attention、stage末尾の共有FFNとmHC writebackを用いて複数playerの位置・rotation・presenceを推定する。
- **`plcs_track_query_reference_model.py`**: 同じarchitectureへcamera-view target frameとreference selectorの6入力contractを追加する。
- **`components/heads.py`**: `PositionHead`/`RotationHead`/`CanonicalPoseHead`。
- **`discriminators/`**: 共有`TransformerSequenceDiscriminator`を`input_dim=5`で構築するPLCS composition factory。

### training/
- **`composition.py` / `runner.py`**: validated configからdatamodule/Lightning lifecycleを外部compositionで一度だけ選択する。
- **`lightning_module.py`**: `PLCSLightningModule`。構築時にmodel-I/O pairを固定し、supervised+canonical+MCMCノイズ+GANを統括。
- **`losses.py`**: `PLCSLoss`/`PLCSLossConfig`。`prepare_inputs()`で検証・canonical変換し、`forward()`はtensor loss termの合算だけを行う。reprojection termは予測position/rotation/canonical poseをworld poseへ統合し、clean 2D poseとのmasked Smooth-L1を全cameraで計算する。
- **`metrics.py`**: `PLCSMetrics`。メートル換算誤差・角度誤差・閾値内accuracyを集計。
- **`mcmc.py`**: `LangevinNoiseInjector`(issue #519)。rotation headのflat saddle脱出用SGLDノイズ注入。
- **`tracking_{matching,losses,metrics,lightning_module}.py`**: clip-level Hungarian matchingとmulti-person固有loss/metrics/payloadを所有し、Lightning stage lifecycleは`tasks/base/training/tracking_lightning_module.py`へ委譲する。

### inference/
- **`predictor.py`**: `PLCSPredictor`。checkpointに対応するadapterを保持し、明示的なvisibility/maskを検証してから推論する。統合consumer向け`predict_multiview_observations()`はmeters/yawのtyped NumPy結果を返す。
- **`tracking_predictor.py`**: track-query専用adapterを保持し、position/rotation/presenceをdecodeする。

### model_io/
- **`contracts.py`**: frame/sequence/multiview/track-query profile、prepared call、standard/tracking decoded prediction、physical predictionの型付き契約。
- **`attention_masks.py`**: standard axial model向けcamera・time attention maskを`padding_mask`から準備する。track-query modelは共有padding utilityを内部で使う。
- **`adapters.py`**: 必須field、dtype、rank、shape、normalized UV、binary mask、view/time capacity、prepared attention tensor、output schemaを`forward`前後の境界で検証するtask-local adapter。
- **`factory.py`**: model variantとadapterを外部compositionで一度だけ選択し、exact model classのpairを固定する唯一のfactory。

### generate_dataset/
- **`config.py`**: standalone generation boundary。共有契約を消費し、run/device/split と生成 worker 用の絶対 path を検証・解決する。
- **`scene_generator.py`**: `SceneGenerator`。source-independentなCOCO-17モーションをコート座標へ配置し、マルチカメラ投影してsceneを構築。
- **`multi_object_scene_generator.py`**: `MultiPersonSceneGenerator`。既存のAMASS/SMPL-H sceneを複数生成し、同一の仮想カメラへ再投影してcanonical multi-person sceneへ合成する。`generation=multi_object` で選択する。
- **`sampling/motion_sampler.py`**: `MotionSampler`。登録済みsource formatをadapter層で共通契約へ変換し、category weightに従ってsampleする。
- **`io/dataset_io.py` / `io/scene_loader.py`**: シーンのnpy/json書き出し・読み込み。
- normalized translation、scene metadata、checkpoint互換性は [`src/utils/README.md`](../../utils/README.md) の単一契約に従い、canonical poseはmetreのまま保持する。
- **`utils/parallel_runner.py`**: CPU専用の並列シーン生成ラッパー。

### visualization/
- **[Web UI利用ガイド](visualization/README.md)**: 閲覧・GPU推論のコピー可能な起動コマンドと操作手順。
- **`review/` / `inference/`**: [データセット閲覧](visualization/review/README.md)と[GT・推論比較Web UI](visualization/inference/README.md)。
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`api/predict.py`**: `predict_scene()`。predictorに固定されたadapterへscene assembly/decodeを委譲する。比較描画のcanonical poseは`visualization.canonical_pose_source=gt|prediction`で選択し、既定ではGTを使う。
- **`contracts.py`**: `PoseRenderScene`。renderer向け最小scene契約。
- **`rendering/scene_renderer.py`**: `PLCSSceneRenderer`。single/multi-personの3D/2D top-down/入力cameraアニメーションとGT・予測比較を描画する。3Dは `src.utils.rendering` の共有プリミティブを利用。style/視点は `visualization.style` / `visualization.view_3d` で設定。
- **`adapters/`**: typed decoded predictionから学習時qualitative描画入力への変換。
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。

### scripts/
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`generate_dataset.py`**: 並列合成データ生成エントリポイント。
- **`extract_gvhmr_motions.py`**: dataset manifestとcamera別foreground ROIからGVHMR motion/raw sidecar/品質recordを抽出する再開可能なエントリポイント。
- **`generate_dataset_samples.py`**: 生成済み各datasetへ層化されたcamera-view GIFとmanifestを作成。
- **`visualize.py`**: 可視化エントリポイント。
- **`analysis/*.py`**: データセット分布・角速度統計・loss dominance・回転誤差サンプル抽出の分析スクリプト群。

### configs/
- 公開data profileは用途ごとに整理している。`singleview_frame`、`singleview_sequence`、`singleview_chunked_sequence`、`multiview_sequence`、`multiview_chunked_sequence`（single_object）、`tracking`、`tracking_chunked`（multi_object）、`singleview_sequence_broadcast`、`multiview_sequence_broadcast`（single_object_broadcast）、`tracking_broadcast`（multi_object_broadcast）、`tracking_camera_view_v2`（multi_object_camera_view_v2）、`multiview_sequence_camera_view_v2`（single_object_camera_view_v2）で、各データセットを固定・chunked・broadcast・camera-viewの用途から重複なく選択できる。
- `tracking_camera_view_v2` はdata profileの選択だけで、Hydraのabsolute overrideにより`court_keypoints=camera_view_v2`と`model=tracking_query_reference`を同時に選択する。その他にmodel(frame/multiview/axial系)・loss(canonical段階別)・training(default/GAN/MCMC)・metrics・motion_sources・simulation/camera/paths(生成用)・visualization・run・analysis の各Hydra設定がある。

## Multi-person tracking

共有の2D observation tracking contract、camera-local slotの意味、overflow、および破壊的migrationの正本は [`src/tasks/base/README.md`](../base/README.md) です。PLCS固有の5観測tensor shapeは `human_kp (B,V,T,Q,17,2)`、`human_vis (B,V,T,Q,17)`、`court_kp (B,V,T,14,2)`、`court_vis (B,V,T,14)`、`padding_mask (B,V,T)` です。`human_vis.any(-1)`がfalseの非padding slotはlearned invisible tokenになりますがattentionには参加します。`detection_gt_index`と`clean_human_kp`/`clean_human_vis`は評価・可視化専用fieldで、モデルへは渡しません。

PLCSの`data.association`は `max_distance=0.08`、`max_missed_frames=8`、`min_reuse_gap_frames=4`、velocity prediction有効、`min_common_keypoints=4`、`cost_reduction=median`、`overflow_policy=error`を初期値とします。Issue #832より前のtracking checkpoint/resultは新しいassociation意味論と互換ではないため、必ず再学習・再評価してください。旧設定とmetricの詳しい移行条件は共有正本を参照してください。

出力は `position (B,T,Q,3)`、`rotation (B,T,Q,2)`、`presence_logits (B,T,Q)` です。教師は独立したtarget lifecycle packingによる `target_position`、`target_rotation`、`target_presence`、`target_instance_id` で、inactive rotationはidentity、instance IDは`-1`です。重ならないbirth/death区間を同じtarget slotへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。

14 court UVは共有Court contractでreference整列した後の先頭14点を使い、`court_vis`で不可視点を0化します。各observation slotのperson keypointsとcourtを連結し、BLCSと同じ`src/utils/models/embeddings/group_tokens.py`の共有`CourtPlayerGroupEmbedding`により1 slot = 1 tokenへ写像します。object temporal後の空間self-attention入力は `(B*T, Q + V, D)` です。M-RoPE座標と第3軸の意味は共有正本を参照してください。

BLCSと共有する各stageは `mHC object temporal -> global spatial(Q+V) -> query temporal` の順で更新し、temporal modeを `CSWA, CSWA, CSWA, Global MHA` のcycleへ固定します。`object_state_valid`を含む全state/attention maskは共有`build_fixed_query_padding_masks()`が`padding_mask`だけから生成します。nested `model.mhc` / `model.cswa`はstrictに検証し、旧`spatial_blocks` / `temporal_blocks` checkpointは自動変換せずstrict load errorとします。

`model=tracking_query`がこの唯一のcanonical architectureを選びます。各attention blockはFFNを持ちません。旧track-query checkpointはarchitectureが異なるためstrict load errorです。

multi-objectのsource全区間保持・可変長・bornの存在数調整は共有正本の「Full-source multi-object lifetimes」に従います。学習時は512〜1024 frame・3〜5 viewをsampleします。chunked設定は`scenes_per_chunk=1000`、`epochs_per_chunk=20`、`prefetch_chunks=5`、`generation_workers=16`、DataLoaderの`num_workers=4`です。

```bash
# 固定train/val/testデータを事前生成
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset \
  generation=multi_object run.output_dir=plcs/multi_object

# 事前生成データで学習
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking

# canonical architecture
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking \
  model=tracking_query

# broadcast two-view tracking（GPUならqueue経由）
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking \
  data=tracking_broadcast

# camera-view reference（別途生成したopt-inデータ、GPUならqueue経由）
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking \
  data=tracking_camera_view_v2

# trainだけon-the-fly chunk生成（val/testは上記の固定データ）
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking_chunked

# train chunkだけACCAD+GVHMRに拡張し、val/testは同じ固定データを維持
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking_chunked \
  motion_sources=accad_gvhmr
```

## Axial reference training recipe

`train_axial_reference` composes the four-corner camera-view data profile,
`multiview_axial_reference` model and `loss=axial_reference`. The model uses
`TemporalDecomposedCanonicalPoseHead`; rotation and wrapped-angle weights are
both 0.1 by default for this recipe. Reprojection has weight 1 with no paired
weight-zero run. Model/data/loss/optimizer values live in these Hydra configs;
this README is the entry point rather than a second copy of the parameter table.
Generation and explicit camera candidate semantics are documented in the shared
contract linked above.

```bash
# Submit through the shared training queue when running on a local GPU.
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_axial_reference
```

The model requires an explicit reference selector and matching provenance.
Reference validity is checked by the paired adapter before compiled forward.
Checkpoint metadata includes the independent `axial_reference` architecture,
target, RoPE and selector markers; physical and track-query checkpoints cannot
be substituted. Position, heading, world-joint and camera transformations follow
the shared reference-frame contract. Direct scene inference requires a stable
`reference_camera_id`; array inference requires explicit reference provenance.

## Foot residual experiment

現行の `plcs_multiview_axial_foot_residual` の構成は Modules の該当項目を参照。学習・同条件の
本番推論比較の結果と当時の実験条件は knowledge の群ノード
[足元疑似位置・幾何埋め込み・残差学習の比較](../../../knowledge/nodes/plcs/000094-group-plcs-foot-residual.md)
に集約している。
`data.sampling_weights` はscene directory内のJSONファイル名を指定する任意項目で、
filtered train splitの全scene名を正の有限重みに対応させる。固定dataset backendのみ対応し、
val/test loaderには適用しない。未指定時は従来のshuffleを使う。


## Triangulation residual

物理コート座標（XY地面・Z上、metre）のCOCO17を、各カメラの
`[p_obs, p_reproj, p_obs-p_reproj, court14, X_init, camera]`とconfidence/maskから補正する。
rootは左右hipの中点、出力はglobal root残差と17関節の相対姿勢残差。
相対残差のhip平均をゼロにし、rootとの重複を除く。既存のSMPL-root/yawモデルとは
異なる入出力契約であり、checkpointを読み替えない。BLCSにはこのモデルを導入しない。

### 責務と設定

- `data/residual_dataset.py`・`residual_datamodule.py`: ACCAD scene読込、motion-source split検査、時間窓・視点選択、loader。GT worldをそのまま読み、既存モデル用の座標変換は適用しない。
- `data/augmentation/observation.py`: 既存の観測augmentation。packageから従来の公開型を再公開する。
- `data/augmentation/residual.py`・`persistent_pose.py`: 四隅＋フェンス付近正面2台の学習camera、Court14の破損と再校正、通常・持続誤検出。旧来の独立camera摂動は使わない。
- `geometry/residual_features.py`: 学習・推論共通の三角測量とraw特徴生成。汎用の投影・DLT・平面camera推定は`src/utils/geometry`を利用する。
- `models/triangulation_residual.py`: camera/time attentionと時間RoPE、ゼロ初期化した2つの残差head。camera順序には依存しない。
- `model_io/residual_contracts.py`・`residual_checkpoint.py`: 入出力とcheckpoint schemaの厳密な検証。
- `training/residual_losses.py`・`residual_metrics.py`・`residual_lightning_module.py`: root/relative/worldの成分別Smooth L1、true-camera/clean-UV再投影、GT速度・骨長、paired診断。実行と構成は既存の`runner.py`・`composition.py`を使う。
- `inference/residual_clip_io.py`・`residual_predictor.py`: 整列済みCourt14と保存cameraを検証して読み、時間windowの予測を融合する。half-turnを二重適用しない。
- `visualization/adapters/residual.py`・`rendering/residual_comparison.py`: 初期姿勢と補正結果の比較。

単一の`configs/train_triangulation_residual.yaml`がdata/model/loss/trainingの各設定を合成する。
数値の正本は各YAML。学習方式のv1/v2、損失legacy/balanced、raw/asinhの切替は廃止し、
[比較実験](../../../knowledge/nodes/plcs/000111-group-geometric-residual-v2.md)で選択した
Court14再校正＋従来の成分別損失＋raw入力に統一する。単一seedでの選択であり、
augmentation各成分の個別効果や実写3D精度を証明するものではない。

### 欠測・校正・学習境界

三角測量には信頼度閾値以上の2視点以上を要求し、無効点はNaNと元のmaskを保存する。
モデル入力のseedだけ時間補間・端点保持を行い、全期間欠測jointは観測由来rootで補う。
rootも全期間観測できなければ失敗し、GTによる補完やcameraへの代用はしない。
全6候補を同じ誤差drawで生成してから実行可能なsubsetを選び、最大8回の再試行でも
GT/window/splitや誤差family/severityは変更しない。

学習cameraはnoisy Court14から焦点距離とR/tを推定し、主点中央・fx=fy・skew/歪み0を仮定する。
画像内の6点以上、非共線性、正depth、地上camera、焦点境界を検査する。
カメラ不確実性の推定やコート対称性の自動解決は行わず、Court14の物理対応は既知とする。
既存の共通実映像pipelineは変更しない。合成誤差分布は実測から校正した値ではない。

GT・true camera・clean UVは損失と診断だけに渡す。loader workerはspawn、OpenCVは1 thread。
validation/testのseed・窓は固定し、最小`val/world_mpjpe_m`のcheckpointでtestする。
実clipには独立3D正解がなく、再投影誤差を絶対3D精度として扱わない。

```bash
# GPUを使う学習・推論は必ず共有training queue経由で実行する。
.venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual \
  paths.data_root=/absolute/repo/data
.venv/bin/python -m src.tasks.plcs.scripts.infer_triangulation_residual \
  --checkpoint /absolute/model.ckpt --clip /absolute/clip_000 \
  --output /absolute/comparison --device cpu
```

checkpointはschema 3のみを通常読込する。採用した旧PLCS Court14/raw checkpoint（schema 1/2）は
次の明示変換で重みと由来を保持した別ファイルにする。これは推論・初期重み用で、旧実験の
optimizer状態を新方式でresumeする変換ではない。旧v1、BLCS、balanced/asinhの再現は
knowledgeに記録した当時のcommit・patchを使う。

```bash
.venv/bin/python -m src.tasks.plcs.scripts.migrate_residual_checkpoint \
  /absolute/old.ckpt /absolute/new.ckpt
```
