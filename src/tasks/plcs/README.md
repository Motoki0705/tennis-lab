# PLCS

2D の人物 pose とコート keypoint から、コート座標系でのプレイヤー `position`/`rotation`（および任意で canonical 3D pose）を推定するタスクです。AMASS/SMPL-H モーションと仮想カメラから学習データを合成する generator、frame/sequence/multiview の各モデル、Lightning 学習、推論、可視化までを一貫して提供します。

## Court-coordinate normalization

正規化の数式と version-to-scale mapping の正本は
[`src/utils/schema/court_normalization.py`](../../utils/schema/court_normalization.py)
です。PLCS の `position` だけを `position_norm = position_court_m / scale_xyz`
で正規化します。`canonical_pose_3d`、`human_kp_3d`、
`position_court_m` は metre のままで、`rotation` / yaw も変更しません。

- `v1`（互換 default）: `scale_xyz = (5.485, 11.885, 1.07) m`
- `v2`: `scale_xyz = (11.885, 11.885, 11.885) m`

すべての Hydra root は `court_coordinate_normalization=v1|v2` を明示的に
compose します。新規 dataset は root と全 scene、新規 checkpoint は root に
version、`scale_xyz`、position/velocity unit を保存します。runtime と artifact の
version または scale が異なる場合、dataset load、resume、evaluation、inference
はいずれも変換せずに error になります。metadata のない既存 artifact は
明示的な `v1` runtime だけで legacy として読めます。shape や値域から version
を推測しません。

`v1` loss は従来の normalized Smooth L1 `beta=1` を維持します。`v2` の
position loss と tracking Hungarian position cost は全軸一様で、default の
物理 transition は `1.0 m`（normalized `1 / 11.885`）です。

version を artifact 名でも区別するため、baseline 用に
`generate_dataset_norm_v1|v2.yaml`、`train_norm_v1|v2.yaml`、
`data/multiview_sequence_norm_v1|v2.yaml` を用意しています。既存 v1 dataset を
上書きせず v2 copy へ materialize する例は次の通りです。

```bash
.venv/bin/python -m src.tasks.base.scripts.materialize_court_coordinate_normalization \
  court_coordinate_normalization=v2 \
  materialization.dataset_kind=plcs \
  materialization.source_dir=data/plcs_broadcast \
  materialization.output_dir=data/plcs_broadcast_norm_v2 \
  materialization.source_normalization_version=v1

.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset \
  --config-name generate_dataset_norm_v2

.venv/bin/python -m src.tasks.plcs.scripts.train \
  --config-name train_norm_v2
```

## Modules

### configuration
- **`configuration.py`**: training・analysis・visualization runtime boundary。共有 contract を消費し、generation package は import しない。
- **`configuration_contracts.py`**: training と standalone generation が共有する path roots と generation component の型付き契約。両 runtime 設定より下位に置き、相互 import を作らない。

### data/
- **`dataset.py`**: `SceneDataset`。sceneをcamera-time基準のcanonical sample(`human_kp`/`court_kp`/`position`/`rotation`等)に変換。
- **`datamodule.py`**: `PLCSDataModule`。model非依存のcanonical `(B,V,T,...)` batchを構築し、profile固有変換は行わない。
- **`augmentation.py`**: `PLCSObservationAugmentation`。UVノイズ・時間jitter・可視性dropout等8段のパイプライン。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`targets.py`**: `build_coco17_world_targets()`。canonical poseまたはAthletePose3DからCOCO17ワールド座標targetを構築。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、物理trackを固定幅lifecycle observation slotへpackingするDataset/DataModule。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成する。val/testは常に`scene_dir`上の固定splitを使う。
- **`tracking_augmentation.py`**: object列を並べ替えず、clean GTを保持したまま観測だけへpose noise/dropout/false-positiveを適用するshape adapter。
- **`types.py`**: `PLCSBatch`/`PLCSSceneMeta` のバッチ・meta契約。

### models/
- **各model module**: 実装classのcanonical import先。package rootは内部classや旧factoryをre-exportしない。
- **`plcs_model.py`**: `PLCSModel`。単視点frame向けdecoder-only Transformer(court+playerトークン)。
- **`plcs_multiview_model.py`**: `PLCSMultiViewModel`。camera×time interleaved RoPEによるmultiview Transformer。
- **`plcs_multiview_axial_model.py`**: `PLCSMultiViewAxialModel`。camera軸/time軸交互self-attention(共有readout)。
- **`plcs_multiview_axial_split_model.py`**: `PLCSMultiViewAxialSplitModel`(issue #518)。rotation/pose trunkを分離。
- **`plcs_multiview_axial_camtoken_model.py`**: `PLCSMultiViewAxialCamTokenModel`(issue #576)。head別に別camera tokenを読む。
- **`plcs_track_query_model.py`**: `PLCSTrackQueryModel`。object ID順のcamera pose観測からclip-localな固定query slotで複数playerの位置・rotation・presenceを推定する。
- **`components/heads.py`**: `PositionHead`/`RotationHead`/`CanonicalPoseHead`。
- **`discriminators/`**: 共有`TransformerSequenceDiscriminator`を`input_dim=5`で構築するPLCS composition factory。

### training/
- **`composition.py` / `runner.py`**: validated configからdatamodule/Lightning lifecycleを外部compositionで一度だけ選択する。
- **`lightning_module.py`**: `PLCSLightningModule`。構築時にmodel-I/O pairを固定し、supervised+canonical+MCMCノイズ+GANを統括。
- **`losses.py`**: `PLCSLoss`/`PLCSLossConfig`。`prepare_inputs()`で検証・canonical変換し、`forward()`はtensor loss termの合算だけを行う。
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
- **`court_coordinate_checkpoint.py`**: checkpoint root の normalization metadata と保存 config を、state restore 前に復元・検証する。

### generate_dataset/
- **`config.py`**: standalone generation boundary。共有契約を消費し、run/device/split と生成 worker 用の絶対 path を検証・解決する。
- **`scene_generator.py`**: `SceneGenerator`。AMASSモーションをコート座標へ変換しマルチカメラ投影してsceneを構築。
- **`multi_object_scene_generator.py`**: `MultiPersonSceneGenerator`。既存のAMASS/SMPL-H sceneを複数生成し、同一の仮想カメラへ再投影してcanonical multi-person sceneへ合成する。`generation=multi_object` で選択する。
- **`sampling/motion_sampler.py`**: `MotionSampler`。AMASS/SMPL-Hモーションの重み付きサンプリングとjoint計算。
- **`io/dataset_io.py` / `io/scene_loader.py`**: シーンのnpy/json書き出し・読み込み。
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
- **`visualize.py`**: 可視化エントリポイント。
- **`analysis/*.py`**: データセット分布・角速度統計・loss dominance・回転誤差サンプル抽出の分析スクリプト群。

### configs/
- model(frame/multiview/axial系サイズ違い)・data(singleview/multiview/chunked)・loss(canonical段階別)・training(default/GAN/MCMC)・metrics・motion_sources・simulation/camera/paths(生成用)・visualization・run・analysis の各Hydra設定。

## Multi-person tracking

モデル入力は `human_kp (B,V,T,Q,17,2)`、`human_vis (B,V,T,Q,17)`、`court_kp (B,V,T,14,2)`、`court_vis (B,V,T,14)`、`padding_mask (B,V,T)` の5 tensorです。`padding_mask=True`だけがattentionから除外されます。各sceneの物理trackはDatasetで固定幅`Q`のlifecycle slotへpackされるため、観測軸とquery軸は常に`P=Q`です。`human_vis.any(-1)`がfalseの非padding slotはlearned invisible tokenになりますがattentionには参加し、時間・camera文脈から更新されます。debug用の`detection_gt_index`は実object由来なら物理instance ID、不可視またはfalse positiveなら`-1`で、モデルへは渡しません。欠損joint UVは0にします。出力は `position (B,T,Q,3)`、`rotation (B,T,Q,2)`、`presence_logits (B,T,Q)` です。教師は `target_position`、`target_rotation`、`target_presence`、`target_instance_id` で、inactive rotationはidentity、instance IDは`-1`です。重ならないbirth/death区間を同じslotへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。

14 court UVはannotation schemaのkeypoint ID順を維持し、`court_vis`で不可視点を0化します。各lifecycle slotのperson keypointsとcourtを連結し、BLCSと同じ`src/utils/models/embeddings/group_tokens.py`の共有`CourtPlayerGroupEmbedding`により1 slot = 1 tokenへ写像します。したがって空間self-attention入力は `(B*T, Q + V*Q, D)` です。M-RoPE `(time,camera,role)` のroleはquery=0、court-player group=1です。

BLCSと共有する各stageは `mHC object temporal -> global spatial(Q+VQ) -> query temporal` の順で更新し、temporal modeを `CSWA, CSWA, CSWA, Global MHA` のcycleへ固定します。`object_state_valid`を含む全state/attention maskは共有`build_fixed_query_padding_masks()`が`padding_mask`だけから生成します。nested `model.mhc` / `model.cswa`はstrictに検証し、旧`spatial_blocks` / `temporal_blocks` checkpointは自動変換せずstrict load errorとします。

multi-object generatorは1024-frame global timelineに3〜10個のAMASS/SMPL-H source subclipを配置し、query再利用gapを含む同時slot占有数を4以下に保ちます。学習時は512〜1024 frame・3〜5 viewをsampleします。chunked設定は`scenes_per_chunk=1000`、`epochs_per_chunk=20`、`prefetch_chunks=5`、`generation_workers=16`、DataLoaderの`num_workers=4`です。

```bash
# 固定train/val/testデータを事前生成
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset \
  generation=multi_object run.output_dir=data/plcs/multi_object

# 事前生成データで学習
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking

# trainだけon-the-fly chunk生成（val/testは上記の固定データ）
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking_chunked
```
