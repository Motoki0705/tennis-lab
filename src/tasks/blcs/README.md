# BLCS (Ball Localization in Court System)

2D のボール観測とコート keypoint から、コート座標系の 3D ボール軌道を推定するタスクです。合成データ生成（物理シミュレーション + マルチカメラ投影）、学習、推論、可視化までを一貫して提供します。

CourtKP20 の version 選択、disk 上の camera-local semantics、model の
reference-frame semantics、metadata と再生成方針は共有の
[`generate_dataset/README.md`](../base/generate_dataset/README.md) を正本とします。
BLCS 固有の差分は、disk の Court 配列が static `(20, 2)` / `(20,)` であり、
reference frame へ position と court-space velocity を同じ proper rotation で
変換する点です。

## Modules

### generate_dataset/
- **`config.py`**: Hydra設定を `GeneratorConfig` に変換する `build_generator_config()`。
- **`scene_generator.py`**: `BLCSSceneGenerator`。1シーン=1ラリーを物理シミュレーションとマルチカメラ投影で生成。
- **`multi_object_scene_generator.py`**: `MultiBallSceneGenerator`。既存の物理ラリーを複数生成し、同一の仮想カメラへ再投影してcanonical multi-ball sceneへ合成する。`generation=multi_object` で選択する。
- **`io/dataset_io.py`**: `BLCSDatasetWriter`/`load_scene()`。シーンのnpy/json入出力。
- normalized position/velocity、scene metadata、checkpoint互換性は [`src/utils/README.md`](../../utils/README.md) の単一契約に従う。
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
- **`tracking_dataset.py` / `tracking_datamodule.py`**: scene読込後にclip/viewをsampleし、physical-width観測をcorruptしてcamera-local trackingしたfixed-Q入力と、独立にpackingしたtargetを構成するDataset/DataModule。通常backendは固定splitを読み、chunked backendだけがtrain sceneを逐次生成する。val/testは常に`scene_dir`上の固定splitを使う。
- **`tracking_augmentation.py`**: query slotを作る前のphysical detectionへdetector noise/dropout/false-positiveを適用し、debug provenanceを更新するadapter。

### models/
- **`blcs_model.py`**: `BLCSModel`。single-view用decoder-only Transformer(court+ballトークン)。
- **`blcs_multiview_axial_model.py`**: `BLCSMultiViewAxialModel`(現行デフォルト)。camera軸/time軸交互self-attention。
- **`blcs_track_query_model.py`**: `BLCSTrackQueryModel`。object streamをviewごとに1 tokenへ圧縮し、FFN-free attention block、`Q+V` spatial attention、stage末尾の共有FFNとmHC writebackを用いて複数ボール軌道とpresenceを推定する。
- **`blcs_track_query_reference_model.py`**: 同じarchitectureへcamera-view target frameとreference selectorの6入力contractを追加する。
- **`components/heads.py`**: constructor時に選択されるposition-only / position+velocity出力module。
- **`components/padding.py`**: 全BLCS modelの公開`padding_mask=True`から、内部validity・attention keep maskを一意に生成する。
- **`components/observation_fusion.py`**: track-query用の固定linear観測融合module。
- **`discriminators/`**: 共有trajectory discriminatorを構築するcanonical factory。

### model_io/
- **`contracts.py`**: trajectory / track-queryのtyped predictionと、学習に必要な全tensorを持つvalidated batch契約。
- **`adapters.py`**: single / axial / track-queryごとの入力検証・versionに対応するmodel call構築・出力decode。v1 / v2のforward契約は共有正本を参照し、attention tensorはadapterで生成しない。
- **`factory.py`**: modelと対応adapterを同時に構築して一度だけbindingするcomposition root。学習・推論loopはmodel名や出力keyを分岐しない。
- **`training.py`**: binding、collate、DataModule、LightningModuleを一括構成する学習runtime root。

### training/
- **`runner.py`**: `BLCSTrainingRunner`。構成済みruntimeを実行し、model固有I/Oを認識しない。
- **`lightning_module.py`**: `BLCSLightningModule`。typed prediction/batchによるsupervised+reprojection+GAN損失を統括。
- **`losses.py`**: `BLCSLoss`。`trajectory_position_loss` + 任意の `reprojection_loss`。微分可能なピンホール投影核は`src/utils/projection/differentiable_projection.py`を共有する。
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
- **`generate_dataset_samples.py`**: 生成済み各datasetへ層化されたcamera-view GIFとmanifestを作成。
- **`train.py`**: 学習エントリポイント(chunked/GAN切替可)。
- **`visualize.py`**: 可視化エントリポイント。

### configs/
- 学習用の公開data profileは10個に固定している。`singleview_sequence` / `multiview_sequence` / `chunked_singleview_sequence` / `chunked_multiview_sequence` は `blcs/single_object`、`singleview_sequence_broadcast` / `multiview_sequence_broadcast` は `blcs/single_object_broadcast`、`tracking` / `tracking_chunked` は `blcs/multi_object`、`tracking_broadcast` は `blcs/multi_object_broadcast`、`tracking_camera_view_v2` は `blcs/multi_object_camera_view_v2` を使う。`chunked_singleview_sequence` は `model=single` と組み合わせる。旧 `chunked_multiview_sequence_bs4/8/16` は廃止した。
- track-queryは`model=tracking_query`と`model=tracking_query_reference`の2 profileだけを公開する。`data=tracking_camera_view_v2`を選ぶと、Hydraのabsolute package override defaultsにより `court_keypoints=camera_view_v2` と `model=tracking_query_reference` が一意に選択される。その他にmodel(single/multiview/axial)・data・training(default/GAN)・loss(default/reprojection/tracking)・physics/rally/camera/targeted_velocity/generator(データ生成)・metrics・visualization・run の各Hydra設定がある。

## Multi-ball tracking

実観測、noise後association、camera-local slot、capacity error、debug metadata、設定と移行の完全な契約は [`tasks/base/README.md`](../base/README.md) を単一の正本とします。

BLCSでは1 detectionを1個のnormalized UV point（`K=1`）として扱い、`min_common_keypoints=1`、`cost_reduction=mean`を必須とします。pre-Q augmentationには`model.num_queries`をsynthetic false-positive容量として明示的に渡しますが、元からvisibleなcarrierは制限しません。既定の`data.association`は`max_distance=0.04`、`max_missed_frames=2`、`min_reuse_gap_frames=4`、velocity prediction有効、overflow errorです。通常版とchunked版は同じassociation configをcomposeします。完全な容量契約は共有正本を参照してください。

modelへ渡すBLCS固有の5観測tensor shapeは `ball_uv (B,V,T,Q,2)`、`ball_vis (B,V,T,Q)`、`court_kp (B,V,T,14,2)`、`court_vis (B,V,T,14)`、`padding_mask (B,V,T)` のままです。`candidate_gt_index`と`clean_ball_uv/clean_ball_vis`はdebug/evaluation専用tensorで、model入力ではありません。

Issue #832より前のtracking checkpoint/resultは新しいassociation学習契約と意味的に互換ではないため、必ず再学習・再評価してください。旧設定とmetricの詳しい移行条件は共有正本を参照してください。

`ball_vis`は観測tokenとlearned invisible tokenの選択だけに使います。非padding位置では`ball_vis=False`のQ tokenもattentionへ参加します。各stageは `mHC object temporal -> global spatial(Q+V) -> query temporal` の順で、temporal modeはconstructor時に `CSWA, CSWA, CSWA, Global` のcycleへ固定されます。nested `model.mhc` / `model.cswa` configはunknown/missing/invalid値をrejectし、`model.cswa.backend=cuda`はextensionが利用不能ならreferenceへfallbackせずconstruction時に失敗します。

`model=tracking_query`がこの唯一のcanonical architectureを選びます。各attention block内のFFNとspatial後の追加query-only FFNはありません。旧track-query checkpointはarchitectureが異なるためstrict load errorです。

出力契約は従来どおり `position (B,T,Q,3)` と `presence_logits (B,T,Q)` です。教師は `target_position (B,T,Q,3)`、`target_presence (B,T,Q)`、`target_instance_id (B,T,Q)` で、inactive IDは`-1`です。旧track-query checkpointはarchitectureが異なるためstrict load errorとなり、自動key migrationやmissing parameter補完は行いません。推論のstrict adapterはexact Qを要求し、`BLCSTrackingPredictor.predict()`だけが`P<Q`入力をzero/invisible tokenでQへpadします。`P>Q`はrejectします。

14 court UVはannotation schemaのkeypoint ID順を維持します。固定linear融合は`court_vis`で不可視点を0化し、Q順の各ball UVと連結して共有`CourtBallGroupEmbedding`により1 query = 1 tokenへ写像します。下流の空間self-attention入力は常に`(B*T, Q + V, D)`です。旧`observation_fusion`、`point_fusion`、`mask_invisible_observations`設定は受理しません。

disk schemaもruntimeと同じ略称に固定し、camera arrayは`cam_{i}_ball_vis.npy`と`cam_{i}_court_kp_vis.npy`を使用します。旧`*_visible.npy`名へのalias/fallbackはありません。観測fieldは`ball_uv`、`ball_vis`、`court_kp`、`court_vis`、`padding_mask`で、`padding_mask=True`だけをpadding極性として使います。version別の追加forward fieldは共有正本を参照してください。内部の`state_valid=True`と`attention_keep_mask=True`はmodel内でのみ導出します。

multi-object generatorは1024-frame global timelineに3〜10個のsource rally subclipを配置し、query再利用gapを含む同時slot占有数を4以下に保ちます。厳格なfull-physics着地点判定でrejectされたsource rallyだけは`generation.maximum_physics_attempts_per_object`の有限budget内で再提案し、予期しない例外やbudget枯渇はそのままhard errorにします。学習時は512〜1024 frame・3〜5 viewをsampleします。chunked設定は`scenes_per_chunk=1000`、`epochs_per_chunk=20`、`prefetch_chunks=5`、`generation_workers=16`、DataLoaderの`num_workers=4`です。

```bash
# 固定train/val/testデータを事前生成
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset \
  generation=multi_object run.output_dir=blcs/multi_object

# 事前生成データで学習
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking

# canonical architecture
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking \
  model=tracking_query

# camera-view reference（別途生成したopt-inデータ、GPU実行はtraining queue経由）
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking \
  data=tracking_camera_view_v2

# trainだけon-the-fly chunk生成（val/testは上記の固定データ）
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking_chunked

# single-view model + on-the-fly chunk generation
.venv/bin/python -m src.tasks.blcs.scripts.train \
  --config-name train_chunked model=single data=chunked_singleview_sequence
```
