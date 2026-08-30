# `src/tasks/base`

各 task パッケージ (`plcs`, `blcs` 等) が再利用する学習・データ入出力・推論・可視化の共通基盤です。抽象クラスは共通のライフサイクル/検証ロジックを持ち、task 固有の部分だけをサブクラスの override ポイントへ委譲する設計です。

## Modules

### Top-level
- **`__init__.py`**: data/training/inferenceの共有APIを再export。

### model_io/
- **`contracts.py`**: task-local adapterが実装するtyped `ModelIOAdapter`、immutableな`ModelCall`、model/adapterをcomposition時に一度だけ検証する`bind_model_io()`/`BoundModelIO`。tensor keyやshapeの意味は各taskが所有する。
- **`tensors.py`**: `TensorSpec`/`require_tensor()`。dtype/rank/fixed dimensionを`forward`前のadapter boundaryで検証する。

### data/
- **`scene_dataset.py`**: `SceneDatasetBase`。シーンディレクトリ読込・window/camera選択・`build_sample()`拡張点を持つDataset基底。`meta.num_frames` は正の整数として必須で、payload長との矛盾は `SceneDataContractError` によりindex作成前に失敗する。
- **`datamodule.py`**: `SceneDirectoryDataModule`。固定 `scene_dir`+split txt を扱うDataModule基底。
- **`chunked_datamodule.py`**: `BaseChunkedDataModule`。trainのみバックグラウンド生成chunkに切替え。
- **`chunk_manager.py`**: `ChunkManager`。chunk生成スレッドのライフサイクル管理。
- **`dataset_writer.py`**: `BaseDatasetWriter`。npy+jsonシーン書き出しの共通実装。
- **`augmentation.py`**: `BaseObservationAugmentation`。augmentation config解析・dispatchガードの共通部分。
- **`canonical_tracking.py`**: tracking sceneのclip/view選択と可変 `(V,T,D)` paddingを担う共通Dataset基盤。
- **`lifecycle_slots.py`**: birth/death区間をinterval coloringで固定query数へ詰め、death後のslot再利用教師を生成。

### training/
- **`lightning_module.py`**: `BaseLightningModule`。optimizer/scheduler構築、明示的なcompile target契約、qualitative/test予測保存の拡張点。
- **`compilation.py`**: `compile_modules()`。primary modelやGAN discriminatorなど、名前付きtargetへidentity/state_dictを保つ`nn.Module.compile()`を適用する。重複参照はobject identityで一度だけ処理し、失敗時にeagerへfallbackしない。
- **`runner.py`**: `BaseTrainingRunner`。configをsingle source of truthとする学習実行の共通フロー。`init_weights`読込後・`Trainer`構築前に全compile targetを処理する。
- **`chunk_rotation_callback.py`**: `ChunkRotationCallback`。epoch終端でchunked datamoduleを回転。
- **`gan_training.py` / `gan_loss.py` / `gan_transition_callback.py`**: 手動最適化ベースのGAN学習共通実装(`LSGANLoss`含む)。
- **`qualitative_callback.py` / `qualitative_saving.py`**: validationサンプルの可視化描画・GIF/画像保存。
- **`losses.py`**: `FocalBCEWithLogitsLoss`。複数taskで重複していた実装を統合。
- **`tracking_lifecycle.py`**: active/inactive/birth/death近傍を重み付けするpresence BCE。
- **`tracking_lightning_module.py`**: BLCS/PLCSに共通するtracking stage dispatch、loss/metric logging、test prediction収集・保存を所有し、task固有adapter/loss/metrics/payloadはhookへ委譲する。
- **`tracking_metrics.py`**: lifecycle segment単位のbirth/death誤差・presence F1・query再利用・ID switch診断。
- **`metric_logging.py`**: train/val/testのheadline allowlist、必須key検証、scalar metricのnumerator/denominatorによるepoch集計を持つstrict contract。

#### Metric visibility and test artifacts

通常logger/progress barにはtaskごとの少数のheadlineと`stage/loss`だけを出し、metric実装が計算するaxis/lifecycle/reference index/loss component等は診断値として分離する。trackingのval/test metricはbatch scalarを平均せず、metricごとの加算可能なnumerator/denominator（frame、segment、有効sequence、reference stratum等）をepoch全体で合算してから比率を求めるため、batch分割とpaddingに依存しない。分母がない診断値は出力せず、必須headlineの分母がないepochは失敗する。test成果物の`metrics.json`はknowledge-control向けheadlineのみ、`diagnostic_metrics.json`はheadlineと重複しない詳細値、`pred_test.npz`は再評価用predictionを保持する。headline欠落や未知stageはfallbackせず失敗する。

## 実2D観測からfixed-Q inputまでの正本契約 (#832)

この節を、BLCS/PLCSの観測association、debug metadata、metric、破壊的migrationに関する唯一の完全な正本とする。task固有READMEはentrypointやtask固有shapeだけを記述し、同じ契約を複製しない。

### 実推論wrapperの出力境界

| 観測 | 実wrapperの出力 | IDと欠測の意味 | synthetic pre-Q observationへの対応 |
|---|---|---|---|
| ball | `BallDetectionModule`はcameraごとに高々1点を返し、`ball_uv`/`ball_uv_px`は`(N,T,2)`、`visibility`/`score`は`(N,T)`。archive境界は`SceneResult.ball_uv (N,T,2)`と`ball_vis (N,T)`。 | ball tracking IDは存在しない。thresholdまたはtrajectory gateで無効になったframeはvisibilityがfalseで、座標とscoreは0。 | visibleな正規化UVを`K=1`の1 detectionとしてcarrier軸`D`へ入れる。dropout/noise/false positiveはこのpre-Q setに適用し、元の単一streamやGT slotを先に`Q`へ割り当てない。 |
| person | `DinoPersonDetector`はBGR `uint8 (H,W,3)`の1 frameからpixel `boxes_xyxy (D,4)`と`scores (D)`を返す。`BotSortAssociator`は各frameについてinteger `id`とpixel `bbx_xyxy (4,)`を返し、`DinoPersonTracker`が選択した各trackを`TrackResult.tracks[id] (F,4)`へ完成させる。ViTPoseはその1 trackのpixel `(center_x,center_y,size) (F,3)`を受け、COCO-17 pixel `(x,y,confidence) (F,17,3)`を返す。 | detector出力とViTPose result自体にはIDがない。BoT-SORT IDは1 video/cameraの1 tracker run内だけで有効で、global/canonical player IDではない。`DinoPersonTracker`は欠けたboxを補間して平滑化するため、完成後のboxからraw detector missingnessを復元・仮定してはならない。 | ViTPoseのpixel座標をimage sizeで正規化し、joint visibilityとともに`K=17`のunordered detection carrierへ入れる。BoT-SORT ID、補間boxの見かけ上の連続性、manual canonical IDはtracking costやmodel input slot labelにしない。 |

multi-camera playerのcanonicalizationは、この境界より後段の`PlayerAssociationModule`/`apply_player_association()`がmanualな時間区間mappingとして所有する。これはcamera-local GVHMR軸をcanonical player軸へ並べ替えるscene reconstruction処理であり、pre-Q observation trackingとは別契約である。同じBoT-SORT番号や同じquery番号が別cameraで現れても同一人物を意味しない。

### pre-Q augmentation、camera-local tracking、fixed Q

共通trackerの入力は、cameraごとのunordered carrier `values (T,D,K,2)`と`visibility (T,D,K)`だけである。座標は有限な正規化UV `[0,1]`とし、物理/GT ID、clean slot、target slot、debug provenance、別cameraのstate、RNGをassociation featureへ入れない。処理順序は次で固定する。

1. 物理観測とdebug provenanceをpre-Q carrierとして抽出する。
2. 設定済みの座標noise、dropout、false positive、時間augmentationをcarrierへ適用する。
3. false positive有効時は`limit_synthetic_false_positive_carriers()`へnoisyなmodel-visible values/visibilityとfalse-positive適用直前のvisibilityを渡し、新規synthetic-only carrierだけを残り`Q`容量へ制限する。
4. `track_camera_observations()`、またはviewごとに独立stateを作る`track_multiview_observations()`で、破損後のvisible UVだけをassociationする。
5. 結果をexact `Q`へ配置し、その後に独立したGT target packingとcollateを行う。

`ObservationTrackingConfig`は`max_distance`、`max_missed_frames`、`min_reuse_gap_frames`、`use_velocity_prediction`、`min_common_keypoints`、`cost_reduction`、literal `overflow_policy: error`をexact keyとして要求する。ball (`K=1`) costは正規化UVのEuclidean distance、pose (`K=17`) costは共通visible jointの距離を設定どおりreduceし、PLCSは最低4 jointのmedianを用いる。gate外または共通joint不足のpairは一致不可である。valid pairから最大cardinality、最小total cost、辞書順 `(slot, canonical_detection_rank)`の順で決定的なone-to-one対応を選ぶ。2観測があればconstant-velocity prediction、なければlast observationをassociation予測に使うが、この予測値はmodel-visible observationを補間しない。

miss中は設定frame数までslot stateを保持し、再出現がgate内なら同じcamera-local slotを再使用する。spare capacityがある通常時は、retire後に`min_reuse_gap_frames`を満たしたslotだけをbirthへ再利用する。現在frameの`visible_count <= Q`なのにbirth用slotが不足するpressure時は、現在matchしたslotを決して奪わず、必要なslot deficitだけを強制再利用する。victimはcooldown/retired slotを`(reusable_after_frame, slot)`順、その後にunmatched retained stateを`(-missed_frames, last_frame, slot)`順で選ぶ。このpressure経路だけは明示的にreuse gapを迂回し、canonical detection順の全birthを、freeと選択済みvictimを合わせた昇順slotへ割り当てる。`TrackingCapacityError`を送出するのは現在frameの`visible_count > Q`だけであり、camera/frame/free-slotを証拠として持つ。切り捨て、暗黙の`Q`拡張、legacy fallbackは行わない。

false-positive容量制限はruntime trackerのoverflow fallbackではなく、augmentationが新しく合成したFP-only carrierだけに適用する明示的なtraining-corruption契約である。false-positive適用直前に1 joint以上visibleだったcarrierはgenuineとして、FP追加後の全model-visible値を変更せず保持する（同一frameのgenuineだけで`Q`を超える場合も保持し、trackerがtyped errorで拒否する）。pre-FP visibilityが全falseで、FP追加後に一部または全部のjointがvisibleになったcarrierだけをsynthetic-onlyと数え、genuine visible carrier後の残り容量まで、noisy visibility maskとvisible UVによるcanonical順で決定的に選ぶ。carrier permutationでcanonicalな選択値は変わらず、rejectしたsynthetic carrierはvaluesを0、visibilityをfalseにする。pre-FP visibilityはこのsynthetic provenance判定専用であり、association cost、gate、slot tie-break、model inputには入れない。

共通の固定幅出力は`TrackedObservations.values (V,T,Q,K,2)`、`visibility (V,T,Q,K)`、`detection_indices (V,T,Q)`である。BLCSは`K=1`を`(V,T,Q,2)/(V,T,Q)`へ、PLCSは`K=17`を`(V,T,Q,17,2)/(V,T,Q,17)`へ渡す。未観測/paddingは0かつvisibility falseで、miss保持中も観測を捏造しない。query番号はcamera-local lifecycle slotであり、別camera、再利用後の別lifecycle、BoT-SORT ID、GT target slotとの恒久的identity対応を持たない。既存head/output rank、mask、V/T collate、prediction-to-target matchingは変更しない。

### debug metadataとpost-#824 metric

`candidate_gt_index`、`detection_gt_index`、`TrackedObservations.detection_indices`/`debug_provenance`、clean observationは評価、renderer、sample inspectionのためだけに保持する。provenanceはassociation後にcarrier indexでgatherし、false positive、dropout、unmatched/paddingを`-1`とする。これらをcost/gate/tie-break、track state、`ModelCall` inputへ渡してはならない。GT target packingもmodel-visible observation packingから独立させる。

input observation trackingとtarget/prediction metric assignmentは別物である。post-#824の`id_switches`はtarget lifecycle内だけで計測し、直前frameの対応が現在も距離gate内ならその1対1対応を優先し、残りをgate付きの決定的な1対1 Hungarian assignmentで対応する。prediction欠落中も同じlifecycleのlast-valid queryを保持し、再対応先が変わったときだけswitchを1件数え、lifecycle境界でresetする。`id_switch_distance=0.05`は正規化court座標単位（約0.59425 m）の必須設定で、近接重複用`duplicate_distance`とは独立である。通常の可視化はmetric専用assignmentを描画しないため、見た目と値は直接対応しない。

#820以後、count系metricはbatch内合計のepoch平均ではなく、評価した全sequenceに対する1 sequence当たり平均で記録する。旧batch集計および各targetが独立に最近傍queryを選んだpre-#824 `id_switches`とは数値互換性がない。保存済みmetricは書き換えず、比較やcheckpoint再選択には現行metricでの再評価が必要である。

### #650系random slotからの破壊的migration

#832は、#650までのGT physical lifecycleを先に`Q`へpackしてからslot randomization/noiseを適用するpipelineを置換する破壊的変更である。`randomize_slots_train`と旧random relabel APIは削除し、`data.association`のexact configだけを受け付ける。旧key、未知key、容量超過を読み替えるcompatibility modeは提供しない。

固定`Q` tensor shapeが同じため旧checkpointが構造上loadできる場合でも、slot identityとaugmentation順序が異なるため、旧checkpointを#832のcontinued trainingや比較へ使用してはならず再学習が必要である。#643/#648/#650のrunは旧associationに加えてpre-#824 metric解釈を含むため、新pipelineの定量baselineとして直接比較しない。同一split、seed、model、学習budgetと現行position/presence/association metricで再実行したrunだけを比較根拠とする。

## Training model compilation

全taskのtraining configは次の共有契約を明示し、標準ではcompileを有効にする。

```yaml
compile:
  enabled: true
  backend: inductor
  mode: default
  fullgraph: false
  dynamic: false
```

`BaseLightningModule.compilation_targets()`は`self.model`をprimary targetとして返す。primary model内で呼ばれる子moduleは個別登録しない。GAN有効時は`ManualGANSupportMixin`が独立して呼ばれる`discriminator`を追加する。将来teacher/studentなどを追加するtaskは`additional_compilation_targets()`で名前付き`nn.Module`を明示する。loss/metricを含む全childの自動探索は禁止する。

標準modeはCUDA Graphsを暗黙に有効化しない`default`とする。`reduce-overhead`と`max-autotune`は明示選択でき、共有Lightning lifecycleがouter batch境界をCUDA Graphsへ通知する。ただしmodel forward内部のgraph breakをまたぐtensor lifetimeはmodel依存であるため、CUDA Graphs modeの互換性は選択したtask/modelで検証する。

`run.dry_run=true`はmodelを構築しないためcompileしない。`run.fast_dev_run=true`は通常どおりcompileする。`run.resume`はin-place compile後も元のstate-dict keyで復元し、`run.init_weights`はweight読込後にcompileする。staged ball detectionのOOM calibrationも同じ設定でprobe modelをcompileする。設定不正、target契約不正、compile失敗は例外として伝播し、暗黙のeager fallbackは行わない。

### inference/
- **`predictor.py`**: genericな`BasePredictor[PredictionT]`。checkpoint/device解決を共有し、predictのdecoded型は選択済みtask adapterが所有する（decoded result内のtensorはCPU）。明示CUDA指定はfallbackせず、availability選択は`auto`だけが行う。

### generate_dataset/
- **`parallel_runner.py`**: `run_parallel_scene_generation()`。`ProcessPoolExecutor`(spawn context)による並列シーン生成fan-out。
- **`timeline_composer.py`**: BLCS/PLCS共通の固定global timelineへsource subclipを配置し、同時存在数とlifecycle metadataを保証。
- **`dataset_samples.py`**: PLCS/BLCS共通の3×3層化選択、時間間引き、GIF検証、`samples/manifest.json`契約。

### visualization/
- **`preview.py`**: dataset previewスクリプト共通helper(`resolve_split_file`/`resolve_sample_indices`等)のcanonical owner。
- **`frames.py`**: 画像ソース読込(`load_rgb_frames`等)。
- **`gif.py`**: `save_gif()`。共通GIF writer。
- **`io.py`**: `BaseSceneBundle`/`resolve_cameras()`。カメラ選択解決の共通ロジック。
- **`layout.py`**: マルチパネル合成ジオメトリ(`compose_row`/`compose_grid`等)。
- **`orchestrator.py`**: `BaseVisualizationRuntimeConfig`/`build_scene_runtime_config()`。可視化オーケストレータ共通スキャフォールド。
- **`style.py`**: `SceneStyleConfig`/`parse_scene_style()`/`parse_view_3d()`。BLCS/PLCS共通の `visualization.style`(theme/影/トレイル/HUD/ミニマップ)と `visualization.view_3d`(共有3D視点)のtyped parse。未知キー・未知テーマはエラー。
