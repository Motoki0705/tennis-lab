# BLCS (Ball Localization in Court System)

2D のボール観測とコート keypoint から、コート座標系の 3D ボール軌道を推定するタスクです。物理 source、学習、推論を提供し、合成 dataset の publication は `src/synthetic_data_generation` の canonical scene pipeline が所有します。

## Modules

### generate_dataset/（physics source）
- **`config.py`**: API server 向けに Hydra 設定を `GeneratorConfig` へ変換。
- **`scene_generator.py`**: `BLCSSceneGenerator`。1シーン=1ラリーを物理シミュレーションとマルチカメラ投影で生成。
- **`multi_object_scene_generator.py`**: `MultiBallSceneGenerator`。既存の物理ラリーを複数生成し、同一の仮想カメラへ再投影してcanonical multi-ball sceneへ合成する。`generation=multi_object` で選択する。
- **`simulation/ball_physics.py`**: `PhysicsConfig`/`BallPhysics`。重力・drag・Magnus・バウンド・ネット/フェンス衝突の物理モデル。
- **`simulation/cell_manager.py`**: `CellManager`。コートを18セルに分割し着地点サンプリング・ショット分類を行う。
- **`simulation/rally_simulator.py`**: `RallySimulator`。サーブ〜リターンの連鎖でラリー全体を生成する中核モジュール。
- **`simulation/targeted_velocity_sampler.py`**: `TargetedVelocitySampler`。指定セルへ着地する初速を解析的+shooting methodで算出。
- **`api_server/`**: シミュレータ探索用FastAPI(`/cells`/`/court_geometry`/`/simulate_shot`)。
- **`webui/`**: 上記APIを叩くNext.jsフロントエンド。

### data/
- **`types.py`**: `BLCSSample`/`BLCSBatch`/`BLCSMultiViewSample`/`BLCSMultiViewBatch` のバッチ契約。
- **`dataset.py`**: `BallTrajectoryDataset`。canonical multiviewサンプルとcanonical collateを提供。
- **`datamodule.py`**: `BLCSDataModule`。composition rootで選択済みのcollateを受け取り、model variantを認識しない。
- **`augmentation.py`**: `BLCSBallObservationAugmentation`。detector誤差を模した8段のUVノイズパイプライン。
- **`tracking_dataset.py` / `tracking_datamodule.py`**: 固定pathのsceneを読み、object観測をscene object IDの昇順で保持したまま、物理trackをlifecycle slotへpackingするDataset/DataModule。
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
- **`predictor.py`**: `BLCSPredictor`。checkpoint内の必須configからmodel/adapter bindingを厳密に復元し、canonical readerまたは明示的なall-view配列からtyped trajectoryを返す。
- **`tracking_predictor.py`**: `BLCSTrackingPredictor`。track-query bindingによりposition、presence logits/probability/判定を一度だけdecodeする。

### scripts/
- **`train.py`**: 固定path datasetを用いる学習エントリポイント。

### configs/
- model(single/multiview/axial・track-queryのサイズ違い)・data・training(default/GAN)・physics/rally/targeted_velocity/generator(source simulation)・metrics・run の各Hydra設定。

## Multi-ball tracking

観測座標は `ball_uv (B,V,T,P,2)`、観測有無は `ball_visible (B,V,T,P)` に一本化し、`ball_candidate_mask` は持ちません。`P` 軸は全camera/frameでscene object IDの昇順に固定し、欠損・dropout・false positiveがあっても列を並べ替えません。debug用の `candidate_gt_index` は観測が実object由来ならその列と同じobject ID、そうでなければ`-1`であり、モデルへは渡しません。scoreやvisibility値を数値特徴へ連結せず、不可視objectはlearned invisible tokenへ置換します。`mask_invisible_observations=true` は不可視tokenをattention keyから除外する対照条件、`false` は`frame_mask` / `view_mask`によるpaddingだけを除外し、不可視tokenを更新可能なmemoryとして使う条件です。出力は `position (B,T,Q,3)` と `presence_logits (B,T,Q)` です。教師は `target_position (B,T,Q,3)`、`target_presence (B,T,Q)`、`target_instance_id (B,T,Q)` で、inactive IDは`-1`です。重ならないbirth/death区間を同じtarget columnへ詰めるため、同一queryはdeath後に別instanceへ再利用できます。

14 court UVはannotation schemaのkeypoint ID順を維持します。`observation_fusion=linear` は`court_vis`で不可視点を0化し、object ID順の各ball UVと連結して共有`CourtBallGroupEmbedding`により1 object = 1 tokenへ写像します。`observation_fusion=point_attention` は各camera/frameについて `[court_0..13, ball_0..P-1]` を32次元tokenへ変換し、court IDとobject ID順のball列を独立軸とする2軸RoPE付きself-attentionで融合します。融合後はball tokenだけをmodel dimへprojectionし、既存の空間・時間attention経路へ渡します。どちらも下流の空間self-attention入力は `(B*T, Q + V*P, D)` です。

canonical scene pipeline は source rally の全frameを保持して global timeline を構成し、固定pathへ transactionally publish します。

```bash
# canonical scene workspaceへBLCS datasetを生成
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  scene_id=B00 targets='[blcs]'

# 事前生成データで学習
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_tracking

```
