# PLCS

2D の人物 pose とコート keypoint から、コート座標系でのプレイヤー `position`/`rotation`（および任意で canonical 3D pose）を推定するタスクです。AMASS/SMPL-H モーションと仮想カメラから学習データを合成する generator、frame/sequence/multiview の各モデル、Lightning 学習、推論、可視化までを一貫して提供します。

## Court keypoint contract

CourtKP20 の version、camera-local disk ordering、reference-frame alignment、
metadata と checkpoint の exact-match 規約は共有正本
[`src/tasks/base/generate_dataset/README.md`](../base/generate_dataset/README.md)
を参照してください。PLCS 固有の差分は、disk の `court_kp_uv` / `court_kp_vis`
がそれぞれ `(T,20,2)` / `(T,20)`、standard sample が整列済み20点、tracking
sample が整列後の先頭14点を使うことです。reference transform は position、
heading、court-space world joints に適用し、player-local `canonical_pose_3d` と
human UV/visibility には適用しません。

## Modules

### configuration
- **`configuration.py`**: training・analysis・visualization runtime boundary。共有 contract を消費し、generation package は import しない。
- **`configuration_contracts.py`**: training と standalone generation が共有する path roots と generation component の型付き契約。両 runtime 設定より下位に置き、相互 import を作らない。

### data/
- **`dataset.py`**: `SceneDataset`。sceneをcamera-time基準のcanonical sample(`human_kp`/`court_kp`/`position`/`rotation`等)に変換。augmentation前の`human_kp_target`/`human_vis_target`と選択camera parameterも保持し、2D reprojection supervisionへ渡す。
- **`datamodule.py`**: `PLCSDataModule`。model非依存のcanonical `(B,V,T,...)` batchを構築し、profile固有変換は行わない。
- **`augmentation.py`**: `PLCSObservationAugmentation`。UVノイズ・時間jitter・可視性dropout等8段のパイプライン。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`targets.py`**: `build_coco17_world_targets()`。canonical poseまたはAthletePose3DからCOCO17ワールド座標targetを構築。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、pose観測をnoise/dropout/false-positiveで破損してからcamera-local trackingにより固定幅`Q`へ変換するDataset/DataModule。target lifecycle packingは観測associationと独立です。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成します。val/testは常に`scene_dir`上の固定splitを使います。
- **`tracking_augmentation.py`**: 固定`Q`へ変換する前の物理幅pose detectionへcorruptionを適用し、false-positive provenanceを`-1`にするadapter。synthetic-only carrierの容量制御は[共有tracking contract](../base/README.md)に委譲する。
- **`types.py`**: `PLCSBatch`/`PLCSSceneMeta` のバッチ・meta契約。

### models/
- **各model module**: 実装classのcanonical import先。package rootは内部classや旧factoryをre-exportしない。
- **`plcs_model.py`**: `PLCSModel`。単視点frame向けdecoder-only Transformer(court+playerトークン)。
- **`plcs_multiview_axial_model.py`**: `PLCSMultiViewAxialModel`。camera軸/time軸交互self-attention(共有readout)。
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
- **`scene_generator.py`**: `SceneGenerator`。AMASSモーションをコート座標へ変換しマルチカメラ投影してsceneを構築。
- **`multi_object_scene_generator.py`**: `MultiPersonSceneGenerator`。既存のAMASS/SMPL-H sceneを複数生成し、同一の仮想カメラへ再投影してcanonical multi-person sceneへ合成する。`generation=multi_object` で選択する。
- **`sampling/motion_sampler.py`**: `MotionSampler`。AMASS/SMPL-Hモーションの重み付きサンプリングとjoint計算。
- **`io/dataset_io.py` / `io/scene_loader.py`**: シーンのnpy/json書き出し・読み込み。
- normalized translation、scene metadata、checkpoint互換性は [`src/utils/README.md`](../../utils/README.md) の単一契約に従い、canonical poseはmetreのまま保持する。
- **`utils/parallel_runner.py`**: CPU専用の並列シーン生成ラッパー。

### visualization/
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`api/predict.py`**: `predict_scene()`。predictorに固定されたadapterへscene assembly/decodeを委譲する。比較描画のcanonical poseは`visualization.canonical_pose_source=gt|prediction`で選択し、既定ではGTを使う。
- **`contracts.py`**: `PoseRenderScene`。renderer向け最小scene契約。
- **`rendering/scene_renderer.py`**: `PLCSSceneRenderer`。single/multi-personの3D/2D top-down/入力cameraアニメーションとGT・予測比較を描画する。3Dは `src.utils.rendering` の共有プリミティブを利用。style/視点は `visualization.style` / `visualization.view_3d` で設定。
- **`adapters/`**: typed decoded predictionから学習時qualitative描画入力への変換。
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。

### scripts/
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`generate_dataset.py`**: 並列合成データ生成エントリポイント。
- **`generate_dataset_samples.py`**: 生成済み各datasetへ層化されたcamera-view GIFとmanifestを作成。
- **`visualize.py`**: 可視化エントリポイント。
- **`analysis/*.py`**: データセット分布・角速度統計・loss dominance・回転誤差サンプル抽出の分析スクリプト群。

### configs/
- 公開data profileは10個に整理している。`singleview_frame`、`singleview_sequence`、`multiview_sequence`、`chunked_multiview_sequence`（single_object）、`tracking`、`tracking_chunked`（multi_object）、`singleview_sequence_broadcast`、`multiview_sequence_broadcast`（single_object_broadcast）、`tracking_broadcast`（multi_object_broadcast）、`tracking_camera_view_v2`（multi_object_camera_view_v2）で、各データセットを固定・chunked・broadcast・camera-viewの用途から重複なく選択できる。
- `tracking_camera_view_v2` はdata profileの選択だけで、Hydraのabsolute overrideにより`court_keypoints=camera_view_v2`と`model=tracking_query_reference`を同時に選択する。その他にmodel(frame/multiview/axial系)・loss(canonical段階別)・training(default/GAN/MCMC)・metrics・motion_sources・simulation/camera/paths(生成用)・visualization・run・analysis の各Hydra設定がある。

## Multi-person tracking

共有の2D observation tracking contract、camera-local slotの意味、overflow、および破壊的migrationの正本は [`src/tasks/base/README.md`](../base/README.md) です。PLCS固有の5観測tensor shapeは `human_kp (B,V,T,Q,17,2)`、`human_vis (B,V,T,Q,17)`、`court_kp (B,V,T,14,2)`、`court_vis (B,V,T,14)`、`padding_mask (B,V,T)` です。`human_vis.any(-1)`がfalseの非padding slotはlearned invisible tokenになりますがattentionには参加します。`detection_gt_index`と`clean_human_kp`/`clean_human_vis`は評価・可視化専用fieldで、モデルへは渡しません。

PLCSの`data.association`は `max_distance=0.08`、`max_missed_frames=8`、`min_reuse_gap_frames=4`、velocity prediction有効、`min_common_keypoints=4`、`cost_reduction=median`、`overflow_policy=error`を初期値とします。Issue #832より前のtracking checkpoint/resultは新しいassociation意味論と互換ではないため、必ず再学習・再評価してください。旧設定とmetricの詳しい移行条件は共有正本を参照してください。

出力は `position (B,T,Q,3)`、`rotation (B,T,Q,2)`、`presence_logits (B,T,Q)` です。教師は独立したtarget lifecycle packingによる `target_position`、`target_rotation`、`target_presence`、`target_instance_id` で、inactive rotationはidentity、instance IDは`-1`です。重ならないbirth/death区間を同じtarget slotへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。

14 court UVは共有Court contractでreference整列した後の先頭14点を使い、`court_vis`で不可視点を0化します。各observation slotのperson keypointsとcourtを連結し、BLCSと同じ`src/utils/models/embeddings/group_tokens.py`の共有`CourtPlayerGroupEmbedding`により1 slot = 1 tokenへ写像します。object temporal後の空間self-attention入力は `(B*T, Q + V, D)` です。M-RoPE座標と第3軸の意味は共有正本を参照してください。

BLCSと共有する各stageは `mHC object temporal -> global spatial(Q+V) -> query temporal` の順で更新し、temporal modeを `CSWA, CSWA, CSWA, Global MHA` のcycleへ固定します。`object_state_valid`を含む全state/attention maskは共有`build_fixed_query_padding_masks()`が`padding_mask`だけから生成します。nested `model.mhc` / `model.cswa`はstrictに検証し、旧`spatial_blocks` / `temporal_blocks` checkpointは自動変換せずstrict load errorとします。

`model=tracking_query`がこの唯一のcanonical architectureを選びます。各attention blockはFFNを持ちません。旧track-query checkpointはarchitectureが異なるためstrict load errorです。

multi-object generatorは1024-frame global timelineに3〜10個のAMASS/SMPL-H source subclipを配置し、query再利用gapを含む同時slot占有数を4以下に保ちます。学習時は512〜1024 frame・3〜5 viewをsampleします。chunked設定は`scenes_per_chunk=1000`、`epochs_per_chunk=20`、`prefetch_chunks=5`、`generation_workers=16`、DataLoaderの`num_workers=4`です。

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
```
