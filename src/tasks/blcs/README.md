# BLCS (Ball Localization in Court System)

2D のボール観測とコート keypoint から、コート座標系の 3D ボール軌道を推定するタスクです。合成データ生成（物理シミュレーション + マルチカメラ投影）、学習、推論、可視化までを一貫して提供します。

## Modules

### generate_dataset/
- **`config.py`**: Hydra設定を `GeneratorConfig` に変換する `build_generator_config()`。
- **`scene_generator.py`**: `BLCSSceneGenerator`。1シーン=1ラリーを物理シミュレーションとマルチカメラ投影で生成。
- **`multi_object_scene_generator.py`**: `MultiBallSceneGenerator`。既存の物理ラリーを複数生成し、同一の仮想カメラへ再投影してcanonical multi-ball sceneへ合成する。`generation=multi_object` で選択する。
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
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、object観測をscene object IDの昇順で保持したまま、物理trackをlifecycle slotへpackingするDataset/DataModule。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成する。val/testは常に`scene_dir`上の固定splitを使う。
- **`tracking_augmentation.py`**: object列を並べ替えず、clean GTを保持したまま観測だけへdetector noise/dropout/false-positiveを適用するshape adapter。

### models/
- **`__init__.py`**: `build_blcs_model(config)`。`model.name` で3実装を切替。
- **`blcs_model.py`**: `BLCSModel`。single-view用decoder-only Transformer(court+ballトークン)。
- **`blcs_multiview_model.py`**: `BLCSMultiViewModel`。クエリのcross-attention+時間self-attentionによる反復更新モデル。
- **`blcs_multiview_axial_model.py`**: `BLCSMultiViewAxialModel`(現行デフォルト)。camera軸/time軸交互self-attention。
- **`blcs_track_query_model.py`**: `BLCSTrackQueryModel`。object ID順のcamera観測からclip-localな固定query slotで複数ボール軌道とpresenceを推定する。
- **`components/heads.py`**: `Trajectory3DHead`/`VelocityHead`。
- **`components/differentiable_projection.py`**: `DifferentiableProjection`。予測3D位置をカメラへ再投影。
- **`discriminators/`**: `BLCSTrajectoryDiscriminator` と工場関数 `build_blcs_discriminator`。

### training/
- **`runner.py`**: `BLCSTrainingRunner`。`data.backend` でdefault/chunked datamoduleを切替。
- **`lightning_module.py`**: `BLCSLightningModule`。supervised+reprojection+GAN損失を統括。
- **`losses.py`**: `BLCSLoss`。`trajectory_position_loss` + 任意の `reprojection_loss`。
- **`metrics.py`**: `BLCSMetrics`。メートル換算L2誤差・閾値内accuracyを集計。
- **`tracking_{matching,losses,metrics,lightning_module}.py`**: clip-level Hungarian matchingによるmulti-ball tracking学習。

### inference/
- **`predictor.py`**: `BLCSPredictor`。`predict(denormalize=True)` でメートル系3D軌道を返す。

### visualization/
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。
- **`adapters/predict_inputs.py`**: single/multiview入力構築。
- **`adapters/render_inputs.py`**: バッチ/出力からGT・予測軌道配列を抽出。
- **`api/predict.py`**: `predict_positions()`。checkpointからメートル単位軌道を返す。
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`rendering/scene_renderer.py`**: `BLCSSceneRenderer`。single/multi-ballの3D/2D/カメラ視点アニメーションとGT・予測比較を描画する。3Dは `src.utils.rendering` の共有プリミティブ(テーマ・レイヤ規約・カメラ・フェード軌道・影・バウンスリング・HUD・ミニマップ)を利用。バウンス表示はmetaのイベント優先、無いときのみ `detect_bounces()` へfallback(`resolve_bounce_frames()`)。style/視点は `visualization.style` / `visualization.view_3d` で設定。

### scripts/
- **`generate_dataset.py`**: 合成データ生成エントリポイント。
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`visualize.py`**: 可視化エントリポイント。

### configs/
- model(single/multiview/axial・track-queryのサイズ違い)・data(single/multiview/chunked)・training(default/chunked/GAN)・physics/rally/camera/targeted_velocity/generator(データ生成)・metrics・visualization・run の各Hydra設定。

## Multi-ball tracking

観測座標は `ball_uv (B,V,T,P,2)`、観測有無は `ball_visible (B,V,T,P)` に一本化し、`ball_candidate_mask` は持ちません。`P` 軸は全camera/frameでscene object IDの昇順に固定し、欠損・dropout・false positiveがあっても列を並べ替えません。debug用の `candidate_gt_index` は観測が実object由来ならその列と同じobject ID、そうでなければ`-1`であり、モデルへは渡しません。scoreやvisibility値を数値特徴へ連結せず、不可視objectはlearned invisible tokenへ置換します。`mask_invisible_observations=true` は不可視tokenをattention keyから除外する対照条件、`false` は`frame_mask` / `view_mask`によるpaddingだけを除外し、不可視tokenを更新可能なmemoryとして使う条件です。出力は `position (B,T,Q,3)` と `presence_logits (B,T,Q)` です。教師は `target_position (B,T,Q,3)`、`target_presence (B,T,Q)`、`target_instance_id (B,T,Q)` で、inactive IDは`-1`です。重ならないbirth/death区間を同じtarget columnへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。

14 court UVはannotation schemaのkeypoint ID順を維持します。`observation_fusion=linear` は`court_vis`で不可視点を0化し、object ID順の各ball UVと連結して共有`CourtBallGroupEmbedding`により1 object = 1 tokenへ写像します。`observation_fusion=point_attention` は各camera/frameについて `[court_0..13, ball_0..P-1]` を32次元tokenへ変換し、court IDとobject ID順のball列を独立軸とする2軸RoPE付きself-attentionで融合します。融合後はball tokenだけをmodel dimへprojectionし、既存の空間・時間attention経路へ渡します。どちらも下流の空間self-attention入力は `(B*T, Q + V*P, D)` です。

multi-object generatorは1024-frame global timelineに3〜10個のsource rally subclipを配置し、query再利用gapを含む同時slot占有数を4以下に保ちます。学習時は512〜1024 frame・3〜5 viewをsampleします。chunked設定は`scenes_per_chunk=1000`、`epochs_per_chunk=20`、`prefetch_chunks=5`、`generation_workers=16`、DataLoaderの`num_workers=4`です。

```bash
# 固定train/val/testデータを事前生成
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset \
  generation=multi_object run.output_dir=data/blcs/multi_object

# 事前生成データで学習
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking

# trainだけon-the-fly chunk生成（val/testは上記の固定データ）
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking_chunked
```
