# PLCS

2D の人物 pose とコート keypoint から、コート座標系でのプレイヤー `position`/`rotation`（および任意で canonical 3D pose）を推定するタスクです。lossless AMASS/ACCAD motion source、各モデル、Lightning 学習、推論を提供し、合成 dataset publication は canonical scene pipeline が所有します。

## Modules

### configuration
- **`configuration.py`**: training・analysis runtime boundary。
- **`configuration_contracts.py`**: task training が共有する path roots の型付き契約。

### data/
- **`dataset.py`**: `SceneDataset`。sceneをcamera-time基準のcanonical sample(`human_kp`/`court_kp`/`position`/`rotation`等)に変換。
- **`datamodule.py`**: `PLCSDataModule`。model非依存のcanonical `(B,V,T,...)` batchを構築し、profile固有変換は行わない。
- **`augmentation.py`**: `PLCSObservationAugmentation`。UVノイズ・時間jitter・可視性dropout等8段のパイプライン。
- **`targets.py`**: `build_coco17_world_targets()`。canonical poseまたはAthletePose3DからCOCO17ワールド座標targetを構築。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: 固定pathのsceneを読み、object観測をscene object IDの昇順で保持したまま、物理trackをlifecycle slotへpackingするDataset/DataModule。
- **`tracking_augmentation.py`**: object列を並べ替えず、clean GTを保持したまま観測だけへpose noise/dropout/false-positiveを適用するshape adapter。
- **`types.py`**: `PLCSBatch` のモデル入力契約。dataset provenanceはcanonical manifestだけが所有する。

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
- **`attention_masks.py`**: axial / track-queryのcamera・time・spatial attention maskとempty-row修復をmodel実行前に準備する。
- **`adapters.py`**: 必須field、dtype、rank、shape、normalized UV、binary mask、view/time capacity、prepared attention tensor、output schemaを`forward`前後の境界で検証するtask-local adapter。
- **`factory.py`**: model variantとadapterを外部compositionで一度だけ選択し、exact model classのpairを固定する唯一のfactory。

### generate_dataset/sampling/
- **`motion_sampler.py`**: canonical PLCS stage が直接消費する lossless `PLCSMotionClip` と ACCAD/AMASS library。

### scripts/
- **`train.py`**: 固定path datasetを用いる学習エントリポイント。
- **`analysis/*.py`**: データセット分布・角速度統計・loss dominanceの分析スクリプト群。

### configs/
- model(frame/multiview/axial系サイズ違い)・data・loss(canonical段階別)・training(default/GAN/MCMC)・metrics・paths・analysis の各Hydra設定。

## Multi-person tracking

観測座標は `human_kp (B,V,T,P,J,2)` のみで、`P` 軸は全camera/frameでscene object IDの昇順に固定し、欠損・dropout・false positiveがあっても列を並べ替えません。debug用の `detection_gt_index` は観測が実object由来ならその列と同じobject ID、そうでなければ`-1`であり、モデルへは渡しません。bbox・keypoint score/visibilityを数値特徴へ連結せず、`detection_mask (B,V,T,P)` がfalseのpersonはlearned invisible tokenへ置換します。`mask_invisible_observations=true` は不可視tokenをattention keyから除外する対照条件、`false` は`frame_mask` / `view_mask`によるpaddingだけを除外し、不可視tokenを更新可能なmemoryとして使う条件です。欠損joint UVは0にします。出力は `position (B,T,Q,3)`、`rotation (B,T,Q,2)`、`presence_logits (B,T,Q)` です。教師は `target_position`、`target_rotation`、`target_presence`、`target_instance_id` で、inactive rotationはidentity、instance IDは`-1`です。重ならないbirth/death区間を同じtarget columnへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。

14 court UVはannotation schemaのkeypoint ID順を維持し、`court_vis`で不可視点を0化します。object ID順の各person keypointsとcourtを連結し、BLCSと同じ`src/utils/models/embeddings/group_tokens.py`の共有`CourtPlayerGroupEmbedding`により1 object = 1 tokenへ写像します。したがって空間self-attention入力は `(B*T, Q + V*P, D)` です。M-RoPE `(time,camera,role)` のroleはquery=0、court-player group=1です。

canonical scene pipeline は各 ACCAD/AMASS clip の全frameを保持し、multi-object global timelineを固定pathへ transactionally publish します。

```bash
# canonical scene workspaceへPLCS datasetを生成
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  scene_id=B00 targets='[plcs]'

# 事前生成データで学習
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_tracking

```
