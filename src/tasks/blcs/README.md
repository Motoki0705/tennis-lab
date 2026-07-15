# BLCS (Ball Localization in Court System)

2D のボール観測とコート keypoint から、コート座標系の 3D ボール軌道を推定するタスクです。合成データ生成（物理シミュレーション + マルチカメラ投影）、学習、推論、可視化までを一貫して提供します。

## Modules

### generate_dataset/
- **`config.py`**: Hydra設定を `GeneratorConfig` に変換する `build_generator_config()`。
- **`scene_generator.py`**: `BLCSSceneGenerator`。1シーン=1ラリーを物理シミュレーションとマルチカメラ投影で生成。
- **`io/dataset_io.py`**: `BLCSDatasetWriter`/`load_scene()`。シーンのnpy/json入出力。
- **`simulation/ball_physics.py`**: `PhysicsConfig`/`BallPhysics`。重力・drag・Magnus・バウンド・ネット/フェンス衝突の物理モデル。
- **`simulation/cell_manager.py`**: `CellManager`。コートを18セルに分割し着地点サンプリング・ショット分類を行う。
- **`simulation/rally_simulator.py`**: `RallySimulator`。サーブ〜リターンの連鎖でラリー全体を生成する中核モジュール。
- **`simulation/targeted_velocity_sampler.py`**: `TargetedVelocitySampler`。指定セルへ着地する初速を解析的+shooting methodで算出。
- **`utils/parallel_runner.py`**: `generate_parallel_scenes()`。CPU専用の並列シーン生成ラッパー。
- **`api_server/`**: シミュレータ探索用FastAPI(`/cells`/`/court_geometry`/`/simulate_shot`)。
- **`webui/`**: 上記APIを叩くNext.jsフロントエンド。

### data/
- **`types.py`**: `BLCSSample`/`BLCSBatch`/`BLCSMultiViewSample`/`BLCSMultiViewBatch` のバッチ契約。
- **`dataset.py`**: `BallTrajectoryDataset`。canonical multiviewサンプルを返し、collate/adapt関数を提供。
- **`datamodule.py`**: `BLCSDataModule`。`input_profile`(`single`/`multiview`)に応じたcollate構築。
- **`augmentation.py`**: `BLCSBallObservationAugmentation`。detector誤差を模した8段のUVノイズパイプライン。
- `data=chunked_multiview_sequence_line_bs4`では、合成court line mapへmap-space augmentationを適用してRANSAC抽出した`court_lines: (V,T,L,4)`だけをcourt入力として返す。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。

### models/
- **`__init__.py`**: `build_blcs_model(config)`。`model.name` で3実装を切替。
- **`blcs_model.py`**: `BLCSModel`。single-view用decoder-only Transformer(court+ballトークン)。
- **`blcs_multiview_model.py`**: `BLCSMultiViewModel`。クエリのcross-attention+時間self-attentionによる反復更新モデル。
- **`blcs_multiview_axial_model.py`**: `BLCSMultiViewAxialModel`(現行デフォルト)。camera軸/time軸交互self-attention。
- 同axial modelの`court_input_type=line`は各cameraを`[court token, ball token]`の2 tokenとして扱い、同じcamera RoPEと異なるtoken-type embeddingを与える。既定の`kp`経路とcheckpoint schemaは変更しない。
- **`components/heads.py`**: `Trajectory3DHead`/`VelocityHead`。
- **`components/differentiable_projection.py`**: `DifferentiableProjection`。予測3D位置をカメラへ再投影。
- **`discriminators/`**: `BLCSTrajectoryDiscriminator` と工場関数 `build_blcs_discriminator`。

### training/
- **`runner.py`**: `BLCSTrainingRunner`。`data.backend` でdefault/chunked datamoduleを切替。
- **`lightning_module.py`**: `BLCSLightningModule`。supervised+reprojection+GAN損失を統括。
- **`losses.py`**: `BLCSLoss`。`trajectory_position_loss` + 任意の `reprojection_loss`。
- **`metrics.py`**: `BLCSMetrics`。メートル換算L2誤差・閾値内accuracyを集計。

### inference/
- **`predictor.py`**: `BLCSPredictor`。`predict(denormalize=True)` でメートル系3D軌道を返す。

### visualization/
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。
- **`adapters/predict_inputs.py`**: single/multiview入力構築。
- **`adapters/render_inputs.py`**: バッチ/出力からGT・予測軌道配列を抽出。
- **`api/predict.py`**: `predict_positions()`。checkpointからメートル単位軌道を返す。
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`rendering/scene_renderer.py`**: `BLCSSceneRenderer`。3D/2D/カメラ視点でのGT・予測比較アニメーション。3Dは `src.utils.rendering` の共有プリミティブ(テーマ・レイヤ規約・カメラ・フェード軌道・影・バウンスリング・HUD・ミニマップ)を利用。バウンス表示はmetaのイベント優先、無いときのみ `detect_bounces()` へfallback(`resolve_bounce_frames()`)。style/視点は `visualization.style` / `visualization.view_3d` で設定。

### scripts/
- **`generate_dataset.py`**: 合成データ生成エントリポイント。
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`visualize.py`**: 可視化エントリポイント。
- **`preview_augmentation.py`**: `preview.court_input_type=kp|line`で、KP augmentationまたはline map劣化 + RANSAC有限線分を比較表示。

### configs/
- model(single/multiview/axialサイズ違い)・data(single/multiview/chunked)・training(default/chunked/GAN)・physics/rally/camera/targeted_velocity/generator(データ生成)・metrics・visualization・run の各Hydra設定。
