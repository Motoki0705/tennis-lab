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
- **`dataset.py`**: `BallTrajectoryDataset`。canonical multiviewサンプルとcanonical collateを提供。
- **`datamodule.py`**: `BLCSDataModule`。composition rootで選択済みのcollateを受け取り、model variantを認識しない。
- **`augmentation.py`**: `BLCSBallObservationAugmentation`。detector誤差を模した8段のUVノイズパイプライン。
- **`chunk_manager.py` / `chunked_datamodule.py`**: バックグラウンドchunk生成によるtrain datamodule。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、object観測をscene object IDの昇順で保持したまま、物理trackをlifecycle slotへpackingするDataset/DataModule。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成する。val/testは常に`scene_dir`上の固定splitを使う。
- **`tracking_augmentation.py`**: object列を並べ替えず、clean GTを保持したまま観測だけへdetector noise/dropout/false-positiveを適用するshape adapter。

### models/
- **`blcs_model.py`**: `BLCSModel`。single-view用decoder-only Transformer(court+ballトークン)。
- **`blcs_multiview_model.py`**: `BLCSMultiViewModel`。クエリのcross-attention+時間self-attentionによる反復更新モデル。
- **`blcs_multiview_axial_model.py`**: `BLCSMultiViewAxialModel`(現行デフォルト)。camera軸/time軸交互self-attention。
- **`blcs_track_query_model.py`**: `BLCSTrackQueryModel`。object ID順のcamera観測からclip-localな固定query slotで複数ボール軌道とpresenceを推定する。
- **`components/heads.py`**: constructor時に選択されるposition-only / position+velocity出力module。
- **`components/observation_fusion.py`**: track-query用に選択済みのlinear / point-attention観測融合module。
- **`components/differentiable_projection.py`**: `DifferentiableProjection`。予測3D位置をカメラへ再投影。
- **`discriminators/`**: 共有trajectory discriminatorを構築するcanonical factory。

### model_io/
- **`contracts.py`**: trajectory / track-queryのtyped predictionと、学習に必要な全tensorを持つvalidated batch契約。
- **`attention_masks.py`**: single / multiview / axial / track-query / point-fusion用のattention maskとempty-row修復をmodel実行前に準備する。
- **`adapters.py`**: single / multiview / axial / track-queryごとの入力検証・prepared attention tensorを含むmodel call構築・出力decode。shape、dtype、device、semantic制約はmodel `forward`前にここで検証する。
- **`factory.py`**: modelと対応adapterを同時に構築して一度だけbindingするcomposition root。学習・推論loopはmodel名や出力keyを分岐しない。
- **`training.py`**: binding、collate、DataModule、LightningModuleを一括構成する学習runtime root。

### training/
- **`runner.py`**: `BLCSTrainingRunner`。構成済みruntimeを実行し、model固有I/Oを認識しない。
- **`lightning_module.py`**: `BLCSLightningModule`。typed prediction/batchによるsupervised+reprojection+GAN損失を統括。
- **`losses.py`**: `BLCSLoss`。`trajectory_position_loss` + 任意の `reprojection_loss`。
- **`metrics.py`**: `BLCSMetrics`。メートル換算L2誤差・閾値内accuracyを集計。
- **`tracking_{matching,losses,metrics,lightning_module}.py`**: clip-level Hungarian matching・forward前のloss term準備・multi-ball固有metrics/payloadを所有し、Lightning stage lifecycleは`tasks/base/training/tracking_lightning_module.py`へ委譲する。

### inference/
- **`predictor.py`**: `BLCSPredictor`。checkpoint内の必須configからmodel/adapter bindingを厳密に復元し、`predict_scene()` / `predict_multiview_arrays()`でtyped trajectoryを返す。
- **`tracking_predictor.py`**: `BLCSTrackingPredictor`。track-query bindingによりposition、presence logits/probability/判定を一度だけdecodeする。

### visualization/
- **`orchestrator.py`**: `run_visualization()`。visualize/predictモードを統括。
- **`api/predict.py`**: `predict_positions()`。checkpointからメートル単位軌道を返す。
- **`io/scene.py`**: `SceneBundle`。シーン読込とカメラ選択。
- **`rendering/scene_renderer.py`**: `BLCSSceneRenderer`。single/multi-ballの3D/2D/カメラ視点アニメーションとGT・予測比較を描画する。3Dは `src.utils.rendering` の共有プリミティブ(テーマ・レイヤ規約・カメラ・フェード軌道・影・バウンスリング・HUD・ミニマップ)を利用。バウンス表示は明示的なscene eventのみを使用し、軌道から意味を推測するfallbackは持たない。style/視点は `visualization.style` / `visualization.view_3d` で設定。

### scripts/
- **`generate_dataset.py`**: 合成データ生成エントリポイント。
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`visualize.py`**: 可視化エントリポイント。

### configs/
- model(single/multiview/axial・track-queryのサイズ違い)・data(single/multiview/chunked)・training(default/chunked/GAN)・physics/rally/camera/targeted_velocity/generator(データ生成)・metrics・visualization・run の各Hydra設定。

## Multi-ball tracking

観測座標は `ball_uv (B,V,T,P,2)`、scoreは`ball_score`、観測有無は `ball_visible`、padding済み候補は`candidate_mask`で明示します。`P` 軸は全camera/frameでscene object IDの昇順に固定し、欠損・dropout・false positiveがあっても列を並べ替えません。debug用の `candidate_gt_index` はモデルへ渡しません。不可視objectはlearned invisible tokenへ置換します。reference cameraで全objectが欠損した時だけP-slot 0をcontextとして残し、隠れた追加tokenは作りません。出力は `position (B,T,Q,3)` と `presence_logits (B,T,Q)` です。

`model=track_query_kp7_reference`は`court_peak_{uv,score,covariance,valid} (B,V,T,7,N,...)`をKP14へ戻さず直接使います。class内はunordered set、ball queryはUV/score/visibilityを持ち、set集約後だけreference roleを加えて既存の`Q + V*P`空間attentionとper-query時間attentionへ渡します。`track_query_kp7_no_reference`は同じtarget orientationでreference deltaだけを外し、`track_query_kp14`はordered-information baselineです。reference camera centerからbaseline距離差1 m以上のviewを選び、target position/velocityのYを一意に反映します。

multi-object generatorは1024-frame global timelineに3〜10個のsource rally subclipを配置し、query再利用gapを含む同時slot占有数を4以下に保ちます。学習時は512〜1024 frame・3〜5 viewをsampleします。chunked設定は`scenes_per_chunk=1000`、`epochs_per_chunk=20`、`prefetch_chunks=5`、`generation_workers=16`、DataLoaderの`num_workers=4`です。

```bash
# 固定train/val/testデータを事前生成
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset \
  generation=multi_object run.output_dir=data/blcs/multi_object

# 事前生成データで学習
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking

# CourtKP7 reference-conditioned
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking \
  model=track_query_kp7_reference

# trainだけon-the-fly chunk生成（val/testは上記の固定データ）
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking_chunked
```
