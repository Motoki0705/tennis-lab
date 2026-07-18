# PLCS

2D の人物 pose とコート keypoint から、コート座標系でのプレイヤー `position`/`rotation`（および任意で canonical 3D pose）を推定するタスクです。AMASS/SMPL-H モーションと仮想カメラから学習データを合成する generator、frame/sequence/multiview の各モデル、Lightning 学習、推論、可視化までを一貫して提供します。

## Modules

### data/
- **`dataset.py`**: `SceneDataset`。sceneをcamera-time基準のcanonical sample(`human_kp`/`court_kp`/`position`/`rotation`等)に変換。
- **`datamodule.py`**: `PLCSDataModule`。`model.io.input_profile`(`frame`/`sequence`/`multiview`)に応じてbatch構築。
- **`augmentation.py`**: `PLCSObservationAugmentation`。UVノイズ・時間jitter・可視性dropout等8段のパイプライン。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`targets.py`**: `build_coco17_world_targets()`。canonical poseまたはAthletePose3DからCOCO17ワールド座標targetを構築。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、物理trackをlifecycle slotへpackingしてからunordered detectionを生成するDataset/DataModule。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成する。val/testは常に`scene_dir`上の固定splitを使う。
- **`tracking_augmentation.py`**: clean GTを保持したままdetectionだけへpose noise/dropout/false-positive/shuffleを適用するshape adapter。
- **`types.py`**: `PLCSBatch`/`PLCSSceneMeta` のバッチ・meta契約。

### models/
- **`__init__.py`**: `build_plcs_model(config)`。5種のモデル実装をdispatch。
- **`plcs_model.py`**: `PLCSModel`。単視点frame向けdecoder-only Transformer(court+playerトークン)。
- **`plcs_multiview_model.py`**: `PLCSMultiViewModel`。camera×time interleaved RoPEによるmultiview Transformer。
- **`plcs_multiview_axial_model.py`**: `PLCSMultiViewAxialModel`。camera軸/time軸交互self-attention(共有readout)。
- **`plcs_multiview_axial_split_model.py`**: `PLCSMultiViewAxialSplitModel`(issue #518)。rotation/pose trunkを分離。
- **`plcs_multiview_axial_camtoken_model.py`**: `PLCSMultiViewAxialCamTokenModel`(issue #576)。head別に別camera tokenを読む。
- **`plcs_track_query_model.py`**: `PLCSTrackQueryModel`。unorderedなcamera pose検出集合からclip-localな固定query slotで複数playerの位置・rotation・presenceを推定する。
- **`components/heads.py`**: `PositionHead`/`RotationHead`/`CanonicalPoseHead`。
- **`discriminators/`**: `PLCSPoseSequenceDiscriminator` と工場関数 `build_plcs_discriminator`。

### training/
- **`runner.py`**: `PLCSTrainingRunner`。`data.backend` でdefault/chunked datamoduleを切替。
- **`lightning_module.py`**: `PLCSLightningModule`。supervised+canonical+MCMCノイズ+GANを統括。
- **`losses.py`**: `PLCSLoss`/`PLCSLossConfig`。position/rotation/canonical/角速度をプラガブルなレジストリで合算。
- **`metrics.py`**: `PLCSMetrics`。メートル換算誤差・角度誤差・閾値内accuracyを集計。
- **`mcmc.py`**: `LangevinNoiseInjector`(issue #519)。rotation headのflat saddle脱出用SGLDノイズ注入。
- **`tracking_{matching,losses,metrics,lightning_module}.py`**: clip-level Hungarian matchingによるmulti-person tracking学習。

### inference/
- **`predictor.py`**: `PLCSPredictor`。`predict(denormalize=True)` で `position_meters`/`yaw_radians` を返す。

### generate_dataset/
- **`config.py`**: `prepare_generation_config()`。パス解決とconfig絶対化。
- **`scene_generator.py`**: `SceneGenerator`。AMASSモーションをコート座標へ変換しマルチカメラ投影してsceneを構築。
- **`multi_object_scene_generator.py`**: `MultiPersonSceneGenerator`。既存のAMASS/SMPL-H sceneを複数生成し、同一の仮想カメラへ再投影してcanonical multi-person sceneへ合成する。`generation=multi_object` で選択する。
- **`sampling/motion_sampler.py`**: `MotionSampler`。AMASS/SMPL-Hモーションの重み付きサンプリングとjoint計算。
- **`io/dataset_io.py` / `io/scene_loader.py`**: シーンのnpy/json書き出し・読み込み。
- **`utils/parallel_runner.py`**: CPU専用の並列シーン生成ラッパー。

### visualization/
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`api/predict.py`**: `predict_scene()`。モデル型に応じてframe/multiview推論を切替。
- **`contracts.py`**: `PoseRenderScene`。renderer向け最小scene契約。
- **`rendering/scene_renderer.py`**: `PLCSSceneRenderer`。single/multi-personの3D/2D top-down/入力cameraアニメーションとGT・予測比較を描画する。3Dは `src.utils.rendering` の共有プリミティブを利用。style/視点は `visualization.style` / `visualization.view_3d` で設定。
- **`adapters/`**: predictor入力構築と学習時qualitative描画用変換。
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。

### scripts/
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`generate_dataset.py`**: 並列合成データ生成エントリポイント。
- **`visualize.py`**: 可視化エントリポイント。
- **`analysis/*.py`**: データセット分布・角速度統計・loss dominance・回転誤差サンプル抽出の分析スクリプト群。

### utils/
- **`pose_geometry.py`**: `src.utils.geometry.court_pose` からのre-export(歴史的importパス維持)。

### configs/
- model(frame/multiview/axial系サイズ違い)・data(singleview/multiview/chunked)・loss(canonical段階別)・training(default/GAN/MCMC)・metrics・motion_sources・simulation/camera/paths(生成用)・visualization・run・analysis の各Hydra設定。

## Multi-person tracking

観測座標は `human_kp (B,V,T,P,J,2)` のみで、bbox・keypoint score/visibilityを数値特徴へ連結しません。`detection_mask (B,V,T,P)` がfalseのpersonはlearned invisible tokenへ置換します。`mask_invisible_observations=true` は不可視tokenをattention keyから除外する対照条件、`false` は`frame_mask` / `view_mask`によるpaddingだけを除外し、不可視tokenを更新可能なmemoryとして使う条件です。欠損joint UVは0にします。出力は `position (B,T,Q,3)`、`rotation (B,T,Q,2)`、`presence_logits (B,T,Q)` です。教師は `target_position`、`target_rotation`、`target_presence`、`target_instance_id` で、inactive rotationはidentity、instance IDは`-1`です。重ならないbirth/death区間を同じtarget columnへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。検出indexはidentityとして扱わず、debug用の `detection_gt_index` はモデルへ渡しません。

14 court UVは`court_vis`で不可視点を0化し、共有point encoderとmean poolingでcameraごとに1 tokenへ写像します。したがって空間self-attention入力は `(B*T, Q + V*(P+1), D)` です。M-RoPE `(time,camera,role)` のroleはquery=0、person=1、court=2で、検出indexやcourt点indexは埋め込みません。court集約は点順序不変で、train時のview単位shuffleにより`far/near`・`left/right`命名不整合にも依存しません。

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
