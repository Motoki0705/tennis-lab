# PLCS

2D の人物 pose とコート keypoint から、コート座標系でのプレイヤー `position`/`rotation`（および任意で canonical 3D pose）を推定するタスクです。AMASS/SMPL-H モーションと仮想カメラから学習データを合成する generator、frame/sequence/multiview の各モデル、Lightning 学習、推論、可視化までを一貫して提供します。

## Modules

### data/
- **`dataset.py`**: `SceneDataset`。sceneをcamera-time基準のcanonical sample(`human_kp`/`court_kp`/`position`/`rotation`等)に変換。
- **`datamodule.py`**: `PLCSDataModule`。`model.io.input_profile`(`frame`/`sequence`/`multiview`)に応じてbatch構築。
- **`augmentation.py`**: `PLCSObservationAugmentation`。UVノイズ・時間jitter・可視性dropout等8段のパイプライン。
- `data=chunked_multiview_sequence_line_bs8`では、合成`court_line_map: (V,T,1,H,W)`を直接返す。既定のline実験configはhuman / line-map augmentationをともに無効化する。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`targets.py`**: `build_coco17_world_targets()`。canonical poseまたはAthletePose3DからCOCO17ワールド座標targetを構築。
- **`types.py`**: `PLCSBatch`/`PLCSSceneMeta` のバッチ・meta契約。

### models/
- **`__init__.py`**: `build_plcs_model(config)`。5種のモデル実装をdispatch。
- **`plcs_model.py`**: `PLCSModel`。単視点frame向けdecoder-only Transformer(court+playerトークン)。
- **`plcs_multiview_model.py`**: `PLCSMultiViewModel`。camera×time interleaved RoPEによるmultiview Transformer。
- **`plcs_multiview_axial_model.py`**: `PLCSMultiViewAxialModel`。camera軸/time軸交互self-attention(共有readout)。
- 同axial modelの`court_input_type=line`は軽量depthwise-separable CNNとsquare-grid poolingでbinary line mapを`num_line_map_tokens`個へ圧縮し、各cameraを`[player, court...]`として扱う。RoPEはtime / camera / typeの3軸で、type座標はplayer=0、全court token=1（位置IDなし）。既定の`kp`経路とcheckpoint schemaは変更しない。
- **`plcs_multiview_axial_split_model.py`**: `PLCSMultiViewAxialSplitModel`(issue #518)。rotation/pose trunkを分離。
- **`plcs_multiview_axial_camtoken_model.py`**: `PLCSMultiViewAxialCamTokenModel`(issue #576)。head別に別camera tokenを読む。
- **`components/heads.py`**: `PositionHead`/`RotationHead`/`CanonicalPoseHead`。
- **`discriminators/`**: `PLCSPoseSequenceDiscriminator` と工場関数 `build_plcs_discriminator`。

### training/
- **`runner.py`**: `PLCSTrainingRunner`。`data.backend` でdefault/chunked datamoduleを切替。
- **`lightning_module.py`**: `PLCSLightningModule`。supervised+canonical+MCMCノイズ+GANを統括。
- **`losses.py`**: `PLCSLoss`/`PLCSLossConfig`。position/rotation/canonical/角速度をプラガブルなレジストリで合算。
- **`metrics.py`**: `PLCSMetrics`。メートル換算誤差・角度誤差・閾値内accuracyを集計。
- **`mcmc.py`**: `LangevinNoiseInjector`(issue #519)。rotation headのflat saddle脱出用SGLDノイズ注入。

### inference/
- **`predictor.py`**: `PLCSPredictor`。`predict(denormalize=True)` で `position_meters`/`yaw_radians` を返す。

### generate_dataset/
- **`config.py`**: `prepare_generation_config()`。パス解決とconfig絶対化。
- **`scene_generator.py`**: `SceneGenerator`。AMASSモーションをコート座標へ変換しマルチカメラ投影してsceneを構築。
- **`sampling/motion_sampler.py`**: `MotionSampler`。AMASS/SMPL-Hモーションの重み付きサンプリングとjoint計算。
- **`io/dataset_io.py` / `io/scene_loader.py`**: シーンのnpy/json書き出し・読み込み。
- **`utils/parallel_runner.py`**: CPU専用の並列シーン生成ラッパー。

### visualization/
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`api/predict.py`**: `predict_scene()`。モデル型に応じてframe/multiview推論を切替。
- **`contracts.py`**: `PoseRenderScene`。renderer向け最小scene契約。
- **`rendering/scene_renderer.py`**: `PLCSSceneRenderer`。3D/2D top-downのGT・予測比較アニメーション。3Dは `src.utils.rendering` の共有プリミティブ(テーマ・レイヤ規約・カメラ・移動トレイル・地面影・HUD・ミニマップ)を利用。ボール軌道は持たないためHUDはフレーム時刻のみ。style/視点は `visualization.style` / `visualization.view_3d` で設定。
- **`adapters/`**: predictor入力構築と学習時qualitative描画用変換。
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。

### scripts/
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`generate_dataset.py`**: 並列合成データ生成エントリポイント。
- **`visualize.py`**: 可視化エントリポイント。
- **`preview_augmentation.py`**: `preview.court_input_type=kp|line`で、KP augmentationまたはCNN入力line mapを比較表示。RANSAC有限線分は旧方式との診断比較用overlay。
- **`analysis/*.py`**: データセット分布・角速度統計・loss dominance・回転誤差サンプル抽出の分析スクリプト群。

### utils/
- **`pose_geometry.py`**: `src.utils.geometry.court_pose` からのre-export(歴史的importパス維持)。

### configs/
- model(frame/multiview/axial系サイズ違い)・data(singleview/multiview/chunked)・loss(canonical段階別)・training(default/GAN/MCMC)・metrics・motion_sources・simulation/camera/paths(生成用)・visualization・run・analysis の各Hydra設定。
