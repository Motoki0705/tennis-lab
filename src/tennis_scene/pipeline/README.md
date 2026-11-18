# 宣言型clip pipeline

各componentは、自身のInput/Output型と必要な成果物schemaを`ComponentIO`で宣言する。
`process(input)`は組み立て済みのInputを受け取る。runnerにモデル名・配列key・カメラ選択を実装しない。

| 場所 | 責務 |
|---|---|
| `components/` | 検出、追跡、pose、人物対応、幾何、身体復元・配置。各IO契約の所有者 |
| `input_assembly/` | 宣言済み成果物のcamera/frame/track照合とInput構築。store全体へ自由に問い合わせない |
| `definition.py` | 実装の選択、入力portとproducerの明示的な接続、モデル資産・設定の束縛 |
| `execution_plan.py` | 複数clipの依存計画、宣言の互換性・循環検査、nodeごとのcache/load判定 |
| `runner.py` | 依存が揃ったnodeの実行・公開、clipの最終化。`ComponentRunner`は単一clipの互換入口 |
| `inference_runtime.py` | 内容ベースの共有キー、モデルの遅延読込・借用・最終需要後の解放、計測hook |
| `model_resources.py` | 複数componentが使うViTPose・外観encoderの共通モデル宣言 |
| `clip_inputs.py` | datasetのclip選択・manifest/media解決、sourceの同期・frame数検証、実行前後の入力照合 |
| `dataset_publication.py` | dataset完成markerの再利用判定、scene契約検証、`annotation.json`の原子的公開と失敗marker |
| `storage/codec.py` | componentが宣言したOutput型をJSONと数値配列へ保存・復元。pickleや動的class importは使わない |
| `storage/clip_store.py` | clipを正本とする永続store、原子的な成果物公開、`scene.json`、メモリcache |
| `storage/scene_export.py` | 統合sceneのimmutableなexportと`scene.json`への採用 |
| `orchestrator.py` | 共通入力部品への動画束縛、標準recipeから1clipの計画を作成・実行・最終化、実行receipt |
| `direct_annotation/` | 白紙注釈request・固定幾何参照と、scene完了・公開だけの再開 |
| `feature_flags.py` | 無効化できる出力機能（`OPTIONAL_FEATURES`）とその依存関係 |

component名の一覧は`contracts.STANDARD_COMPONENTS`が正本で、`pipeline.yaml`の`execution`はその全てを1回ずつ持つ。
`ball_detection.method=model`が既定。`gpt_annotation`の設定・分離したreceipt・復旧・後続受入は
[一連実行のREADME](direct_annotation/README.ja.md)を正本とする。

## 実行計画と入口

`ExecutionPlan`は`ClipExecution(clip_id, nodes, store, overwrite, targets)`の列を受け取り、
nodeを`(clip_id, node名)`で識別する。bindingsは同じclip内のproducerだけを参照し、
異なるclipに同じnode名があってもstoreと依存参照は混ざらない。clip IDとstoreの重複は拒否する。
`targets`を指定すると、その依存閉包だけを実行する。

`PlanExecutor.run()`は依存が揃ったnodeを順次処理する。
`select_ready(ready_keys) -> key`で、その時点で実行可能なnodeの選択方針を差し替えられる。
既定はruntimeが常駐中のモデルを使うready nodeを優先し、次にモデル不要のnode、最後に宣言順を使う。
同順位では追加で必要なモデルが少ないnodeを先に選ぶ。依存の揃わないnodeは選ばない。
cache/load判定は依存artifactが確定してから1回だけ行い、完了したnodeはoverwriteでも再実行しない。
外側から`run(targets=...)`を繰り返してschedulerを作る必要はない。
計画中のclip一覧・実行順・選択方針はidentityに加えず、componentの設定全体
（モデル種別・前処理を含む）・資産・実装・source・上流参照の既存identityを保持する。

`finalize_clip(clip_id)`は、そのclipの必要nodeが完了した時点でsceneをexportする。
`run(on_clip_complete=callback)`はclipが完成した直後に1回通知するため、
callbackから最終化とdataset公開を呼び、後続clipの処理を待たずに公開できる。
通知はcallbackを呼ぶ前に`completion_attempts`へ記録し、成功・失敗とエラー情報を
`completion_results[clip_id]`（`ClipCompletionResult`）へ保存する。
callbackの例外はそのまま呼出し側へ伝わる。同じexecutorを再開しても、失敗したcallbackは再通知しない。
呼出し側はnode完了と公開成否を分けて読み、失敗clipの処理と独立した他clipの再開を判断する。
他clipの完了を待たず、失敗・未処理のclipは最終化しない。モデルの共有・寿命管理やbatch化は実行計画の責務に含めない。
既存の`ComponentRunner(nodes, store)`・`run(targets=...)`・`output(name)`は1clipの同じ計画へ委譲する。

単一動画入口は`prepare_clip_source`で実mediaの時間軸を検証し、入力hashを実行後・export前に照合する。
dataset入口は`select_dataset_clips`・`resolve_dataset_clip`で選択・manifestとrecordの時間軸・mediaを検証し、
同じ単一clip入口を呼ぶ。完成markerの呼出し順序と失敗継続は
[generate_dataset](../generate_dataset/README.md)を参照する。共通関数は複数clip入口からも個別に呼び出せる。

## モデルの所有と寿命

推論を持つcomponentは`InferenceComponent`を実装し、`model_specs()`でaliasと`ModelSpec`を宣言する。
`process_with_models(inputs, models)`は`models.borrow(alias)`で借りる。宣言やcache判定ではモデルを起動しない。
通常の`process(inputs)`も、同じruntimeを1nodeの寿命で使う。componentには重みを保持しない。

`ModelSpec`はモデル種別・architecture/encoder・device・dtype・前処理/推論設定と、名前付き資産path、
生成・初期化・解放のcallbackを持つ。`key`は資産のSHA-256と設定のcanonical JSONを含み、
clip/camera IDと資産pathを含めない。同じ内容の別pathは共有でき、異なる重み・設定は共有しない。
checkpoint内のarchitectureは重み全体の内容identityにも含まれる。
runtimeはnodeの成果物settingsに`file_identity`で束縛したSHAを`expected_assets`として受け取り、
初回借用時もそのSHAと照合する。計画作成後・初回借用前の置換、宣言に無い資産の追加、欠落は公開前に停止する。
開始した計画内でキーを固定し、借用時・初期化直後の資産変更も停止する。
Courtはcheckpoint metadataをCPUで読み、同じresolverで解決した外部DINOv3 backboneとrepository内の
全Python source（相対名と内容SHA）を共有キーと成果物settingsの`external_assets`に含める。
通常clipの完成markerを照合するpublication identityにも同じCourt資産を含め、外部資産の変更を再利用で見逃さない。
cache/loadでも資産の宣言・metadata検証は行うが、推論モデルは構築しない。
順序選択の未常駐モデルの宣言は、種類・設定・資産pathだけで必要数を見積もり、資産を要求しない。
初回借用と、常駐モデルに対する同条件の残り需要を調べる段階で内容キーを解決する。
分岐により借用されない任意モデルの資産が無くても、cache/loadや他モデルの処理を妨げない。
実際に借用する場合の資産欠落は停止し、別モデルへ置き換えない。

`PlanExecutor.run()`は今回のtargetsの未完了nodeにruntimeを1つ作る。
最初の借用で初期化し、そのキーを宣言する最後のnodeが完了した直後に解放する。
cache hitのnodeも需要を消費するが借用しない。明示的な`load` nodeには最初からモデル需要がない。
全cache hit・全loadならモデルの生成・初期化・解放イベントは0となる。
例外・KeyboardInterrupt・公開失敗でもcontext終了時に残るモデルを解放する。
原因例外・抑制されたcontext・例外groupを含む失敗tracebackの完了済みframeからlocal参照を除去する。
例外の構造とstack位置は診断用に保持する。nodeの参照と成功statusは解放・hook完了後に確定し、
解放に失敗したnodeでは依存先とclip完成通知を進めない。同じexecutorを再開してもfailed状態を保持する。

共有対象はCourt、DINO、Ball、ViTPose、CLIP（追跡・選別の画像推論経路・人物対応）、HMR2、GVHMR、
身体配置のSMPL-X/変換行列/関節・root regressor。tracker・AFLink・時系列入力は共有モデルへ混ぜない。
追跡はViTPoseとCLIP、現行GVHMR nodeはHMR2とGVHMRを同時に必要とする。
依存に従ってモデルの段階をまとめ、全モデルを常時GPUへ載せない。
途中targetsで`run()`を分けるとcontextごとに解放し、次の実行で必要なモデルは新しいpassとして読み込む。
単独の`execute(key)`も1nodeで寿命を閉じる。処理済みartifactは再計算しない。

計測は任意の`model_event(action, model, *, weights, device, dtype, instance)` context-manager hookへ渡す。
`PlanExecutor`・`ComponentRunner`・`TennisSceneOrchestrator`（`from_config` / `from_runtime_config`を含む）の
`model_event=`から指定でき、実際の生成/初期化と解放だけを囲む。借用回数をload回数として数えない。
hook失敗時も後始末を実行し、エラーは呼出し側へ返す。
C1のrecorder統合時はその`model_event`を渡し、管理下のモデルで下位の`model_action`/`model_release`と
二重計数しないよう、runtime境界を記録の所有者にする。


## 処理単位

`court_detection`・`person_detection`・`person_tracking`・`player_selection`・`pose_estimation`・`ball_detection`はcameraごとに独立する。
既定の `court_kp.mode=clip` では、`court_detection`は各cameraのframe 0だけをKP＋LINE共同推定し、`court_observations` schema v2で保存する。
明示的な `video_shared` は[動画共有コート成果物](video_court/README.ja.md)をbuild/loadし、同じvideo・cameraの共通Hとdrift診断をschema v3で入力する。
`court_calibration`はこの1frameから初期校正・ROIを作り、固定cameraのコート座標を全frameへ明示的にbroadcastする。
`observed_frame_indices=[0]`と`temporal_policy`を保存し、他frameでモデルを実行したとは扱わない。
人物検出は校正成否に依存せず全画面で実行する。未校正cameraの選手選別は理由付きの空結果を保存する。

```text
動画 → court_detection → court_calibration
動画 → person_detection → person_tracking
track＋court → player_selection → pose_estimation
動画 → ball_detection → ball_points
ball_points＋court → court_side（ballだけのhalf-turn仮説検定）
選別group＋court＋side（＋動画のcrop） → player_association
人物対応＋side＋2D観測 → camera_alignment
人物観測＋camera → player_triangulation
ball_points＋camera → ball_triangulation
人物対応＋2D観測＋camera → body_view_selection → gvhmr
GVHMRパラメータ＋3D関節 → body_placement → scene_assembly
```

人物検出はDINOに固定し、標準設定は**COCO DINOの全画面、score ≥ 0.30、入力800/1333**（#964の2026-09-30判断）。
ROI gateを置かず、選手以外も2D候補として保存する。#937は
`people_models.dino_checkpoint=player_detection/chat-player-v1-e8-best-pr937.pth`を明示した場合に使用できる。
指定重みが無い・DINO形式でない場合は停止し、別の重みを選び直さない。
`person_detections` v2のcheckpoint hash・全画面scopeがartifact identityと下流の依存参照を変える。
旧ROI検出・pose・文脈を新しい経路の成果物として再利用できない。

標準追跡は`person_tracking.method=strongsort_pp_pose`だけを受け付ける。StrongSORT++＋pose/CLIPのrun 10固定profileを
productionと#935用の[共通入口](../../tasks/person_tracking/README.md)で処理する。
`person_tracks` v5は元検出box/row・pose・CLIPを持ち、AFLinkで結んだ元IDも保存する。
GSIの`reconstruction.boxes/interpolated`は別配列で、`observed`は実検出のみ。
選別・人物対応・pose出力はsynthetic boxを観測へ入れない。旧方式はtaskの比較benchmarkで使用する。
重み・方式・特徴契約のエラーは停止し、別方式へ戻さない。
追跡の進捗は`person_tracking/<camera ID>`ごとに、空検出を含む処理済みframe数と総数を表示する。
端末では1本のbar、リダイレクト時は開始・最大30秒に1回の途中経過・最後の実数を出す。
この数はframe単位の処理を表し、その後のAFLink/GSI処理の完了を意味しない。
Pythonからの呼び出しでは`PersonTrackingModule(..., show_progress=False)`で表示を抑制できる。

`person_tracking.aflink_checkpoint`はcheckpoint root相対の`person_tracking/AFLink_epoch20.pth`。
公開重みをcheckpoint root内へ配置し、この設定でroot相対の場所を明示する。
共通PathResolverに従い、絶対パスやroot外へのsymlinkは受け付けない。hashはAFLink readerが検証する。
**AFLink重みの利用条件は未確認**で、当面使用するユーザー判断と後日の再学習判断は
[出自/制約の正本](../../tasks/person_tracking/strongsort_NOTICE.md)を参照。

`people_models.merge_duplicate_person_boxes=false`が既定。trueでは検出直後、追跡/特徴生成前に
IoU>=.8をgreedyに統合する。高scoreを残し、同点は元row順。boxを平均せず、推移的な連結もしない。
v2の`source_rows`は間引き前のclip/camera通番、`duplicate_merges`はframe・keep/drop row・両score・IoU。
旧検出v1/追跡v4/選別v1はloadせず再生成する。cache identityには統合設定と各重みを含める。

`player_selection` は校正z=0の足元から選手候補を選ぶ。固定規則の正本は
[`court_linking.py`](../../tasks/person_tracking/court_linking.py)のdocstringと`LinkingConfig`。
連結後のdistinct core滞在で選別し、`person_observations.max_tracks_per_camera`は最後のgroup上限だけに使う。
CLIP-ReIDは既定on。欠測は明記し、encoder/重みエラーを幾何だけの成功に変えない。
`selected_player_tracks` v2の`selected`は元track軸の全実観測を保持し、領域外・無効足元を削らない。
元boxは参照先`person_tracks`に保存され、`raw_track_ids`で対応する。
poseと既存v3人物対応へ渡す`tracks`は1 group/frameの時系列で、handoff重複だけを小さい元ID優先でまとめる。
`origin_rows`と連結診断に出自を保存する。group IDとraw tracker IDを混同しない。
追跡前のViTPose/CLIPを元rowでgroup軸へ移し替える。欠落を実観測として補間しない。

`player_association`（`components/identity.py`、`person_identities` schema version 3）は
[src/tasks/player_association](../../tasks/player_association/README.md)の対応付けを、校正済みcameraのtrackのboxと
`court_side`が決めたsideに適用する。方式のパラメータはtaskの採用済みdefault YAMLを読み、
シングルス/ダブルスは`player_association.players_per_side`で明示する。CLIPの重みを資産identityに含める。
比較用の旧尺度・別encoderはtaskの比較APIから使用する。
player IDはframeごとで（`player_ids` (V, D, T)）、1本のtrackがID switchの前後で別の人物を持てる。
決まらないclipは`ReconstructionUnavailable`（reason `player_association_<停止理由>`、全scoreと途中の決定）で停止する。

`court_side`は[src/tasks/court_side](../../tasks/court_side/README.md)の仮説検定を、校正済みcameraと単一ball観測の
~30fps格子に適用する（schema version 3）。成果物は採用したhalf-turn、全仮説のcost/support/使用frame数、marginを持つ。
決まらないclipは`ReconstructionUnavailable`（reason `court_side_<停止理由>`、仮説ごとの証跡）で停止する。
ball検出を無効にした構成では実行できず、definition構築時に停止する（`load`は可）。
`camera_alignment`は決まったsideを1仮説として、人物とballの観測で絶対的な整合性だけを再検証する。

`body_view_selection`は人物ごとに観測frame数、平均信頼度、camera ID順で1viewを選び、実行区間を成果物化する。
`gvhmr`はHMR画像特徴とGVHMRだけを実行する。SMPLのmesh/COCO17変換と位置・yaw・scaleの配置は`body_placement`が担当する。

## ball検出証拠

`ball_detection` は `ball_detections` schema v2を保存する。
`BallDetectionOutput.uv_px/confidence/observed` は従来どおり閾値・trajectory gate後の単一観測。
`evidence: BallHeatmapEvidence` はgate前の
[taskのheatmap・候補・局所patch](../../tasks/ball_detection/README.md#検出証拠の出力契約)を
全source frameに対して持つ。`candidate_uv_px` はsource動画の画素座標
（taskのgrid正規化座標にsourceのW−1/H−1を掛ける）。
native格子の解像度は `heatmaps.shape[-2:]`、元動画サイズは `source_size_wh` が正本。

重複窓は `overlap_aggregation` で単一点と**全証拠を同じ窓から**採用し、
`selected_window_start/selected_time_index` に出自を保存する。
最大scoreが同点なら後の窓を採用する。短いclipで入力末尾を反復した場合も、
出力は元frameを1度だけ持つ。strideが窓より長い、またはdropした末尾などで全frameを
覆えない設定は停止し、未処理のframeを負例や空heatmapとして埋めない。

`nearest_window_centre_then_earlier_start`は中心距離が最小の窓、
同点なら早い開始位置を採用する明示的なpolicy。このpolicyは`tail_policy=backfill`を要求し、
短clipのRGB反復を拒否する。標準sceneはこの中心距離policyを使用する。

モデル実行では `evidence` は必須。注釈importと無効な検出器は `None` を明示し、
`score_semantics` で区別する。
下流のside・幾何・三角測量は、観測maskを保持する `ball_points` を使う。

保存v3注釈と固定幾何を使う別経路は[保存済み入力](imports/README.ja.md)を参照する。
taskの位置利用可・直接観測を分離した専用schemaを使用し、通常モデル経路の観測限定契約を維持する。
v1 artifactの自動補完は行わず、executeで再生成、loadはschema不一致で停止する。

## X3D/CSN＋DPT T32の明示選択

`ball_detection.model_type` は必須で、既定は `conv_next_unet`。
`pipeline_x3d_dpt.yaml` は共通事前学習10kのX3D/CSN＋DPTを選ぶoverride。
`pipeline.yaml` の既定checkpointは変わらない。
学習時validationとの履歴一致は、人間が現状の証拠で受入れ（2026-11-16）。
固定した104frame中BF16の5frameが事前許容差を超えた数値は維持している。
判断の出典・実動画比較・残件は[受入記録](../../../knowledge/nodes/tennis_scene/000043-run-scene-x3d-dpt-20261116.md)を参照。

元のweights-only artifactはRGB/MDD契約とTを持つが解像度を保存していない。
scene用artifactは、固定validation manifestの全clipが720×1280であることを確認して
`scene_input_contract` を付加したものを使う。モデルstateは元checkpointと厳密一致する。
取得・SHA照合・metadata追加・validation閾値再計算の再現手順:

```bash
# Driveから取得した完了snapshotをoutputs/の同じ相対pathへ置いてから実行する。
PYTHONPATH=. .venv/bin/python knowledge/runs/run-scene-x3d-dpt-20261116/prepare.py \
  --snapshot outputs/ball_detection/train/capacity_scaling/s42-20261103-main/common-dpt-T32/completed-00010000-dc030faf \
  --checkpoint ckpt/ball_detection/x3d-csn-dpt-T32-common-u10000-scene.pt \
  --report outputs/tennis_scene/evaluate/x3d-dpt/export-calibration.json

# GPU実行は共有training queueへ登録する。動画はDATA root相対。
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline --config-name pipeline_x3d_dpt \
  'video_paths=[match/cam0.mp4,match/cam1.mp4,match/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]'
```

uint8 RGBをOpenCVのlinear補間で720×1280へresizeし、各T32窓のmodel I/O境界で
RGB＋MDDの5chをFP32で作る。ImageNet正規化は加えず、各窓先頭のMDDだけを0にする。
stride16・実frameの末尾backfill・中心距離最小（同点は早い窓）を固定し、短いclipの反復padは拒否する。
窓batchの形は既存と同じ `(B,T,C,H,W)`、dtypeはuint8。
precisionは明示のbf16、batch_sizeは1、`compile_mode=default`で学習時のvalidationに合わせる。
初回はcompileの準備時間を含む。fp32・`compile_mode=off`は明示overrideで選べるが、
学習時BF16＋compile(default)とは別の数値実行条件になる。
メモリ不足でprecision・batch・解像度を自動変更しない。
native /4 heatmap（180×320）と候補は既存のevidence/gate/保存へ渡し、
格子正規化座標に元動画の `(W−1,H−1)` を掛けてpixelへ戻す。

存在閾値はvalidationでbalanced `(FPR+FNR)/2` を最小化した **0.5380124449729919**。
同点は低FPR、次に高閾値を選び、同じscoreのframe群は分割しない。
observed 85,473、確認済み不在31,885に対してFPR 16.735%、FNR 17.099%。
maskedは除外し、位置誤差条件は含めない。同じvalidationで較正・測定した値で、独立testではない。
実行時の再較正は行わない。model_type・channels・T・解像度・正規化・窓policyの不一致、
空間metadataのない元artifact、strict復元エラーは停止する。
設定と重みの内容はcomponent cache identityに含まれる。

## ボール点と欠損

`ball_detection → ball_points → court_side / camera_alignment / ball_triangulation` を使う。
`ball_points` schema v4は、`ball_smoothing`設定で局所的な2次運動consensusによる外れ値除去と、
confidence重み付きの頑健な2次LOESSをCPUで実行する。時間窓は元動画のFPS、
距離閾値は画像の対角長に合わせる。短い急な方向転換は近傍の実測移動量で再検査する。
`observed`は採用した**実検出だけ**で、confidenceも元検出値を保持する。
`smoothing`には`source_observed`・`accepted`・`rejected`・`interpolated`・`smoothed`・
`consensus_supported`・`restored_turn`を個別に保存する。補間座標はこの診断だけに保持し、
side・alignment・三角測量へ渡す`uv_px/confidence`は欠損をゼロ、maskをfalseにする。
長い欠損と両端は埋めず、局所支持の少ない実観測は元の座標と`smoothed=false`を明示して保持する。
有効な観測が2view未満の時刻は3Dでも欠損のまま保持する。
旧v1/v2/v3 storeは現行pipelineへload不可。executeで再生成する。galleryは履歴v3とv4を区別して表示する。
`ball_smoothing.enabled=false`は実観測だけを返す明示的な比較設定。2D Refinerのbundleは不要。

保存済みREADME sceneの比較では、通常runnerのcode fingerprint検証を迂回して旧artifactを
新しい実行identityへ付け替えない。カメラ・人物を固定する実験用replayを別出力として保存し、
元storeの依存鎖・checksum、元動画の明示的なpath対応、再実行したcomponentと固定した配列を記録する。
コマンド・定量比較・表示検証は[CPU replay記録](../../../knowledge/nodes/tennis_scene/000031-run-scene-ball-loess-20261026.md)を参照。


## 成果物

構造化clipでは`<clip>/annotations/tennis_scene/`をstoreとし、呼び出し側が`store_root`で明示する。
単独動画の入口（`store_root=None`）は`cache.directory/<source digest>`を使う。
storeの場所を動画パスの形から推測しない。`cache.directory`はrun IDを含まないARTIFACT pathで、`cache.source=load`で再開できる。

```text
annotations/tennis_scene/
├── scene.json
├── components/<component>/<artifact ID>/<component>.json
├── components/<component>/<artifact ID>/array_0000.npy
├── exports/<scene artifact ID>/scene.npz
├── exports/<scene artifact ID>/scene.metadata.json
└── run.json
```

component JSONは出力schema/version、入力artifact参照、設定・資産・実装識別、source/時間軸、配列参照とchecksum、出自を記録する。
camera scopeはnode名（例`ball_detection/cam0`）と実行identityに含む。
資産identityは有効な機能が読むcheckpointのSHA-256で、ファイルが無ければdefinition構築時に停止する。
大配列は`.npy`へ分離する。読み込みはmmapを使い、メモリにある同じartifactを再利用する。
配列を含む保存・復元結果を下流へ渡すため、同じ実行内と再起動後で型や軸順が変わらない。

出力は一時directoryで完了してから公開し、clip単位のlockを取って`scene.json`を更新する。
中断途中の出力を下流へ渡さず、既存の完成artifactを上書きしない。
`scene.json`の`artifacts`は採用版、`exports.scene`は完成した統合出力とその入力版を指す。
採用componentが差し替わった時点で旧exportへの公開参照を外す。既存のimmutableなexportファイルは保持し、
`scene.json`から統合sceneを読む際にも上流artifactの依存鎖が現採用版と一致するか検証する。
`scene.npz`は派生した統合結果で、component間の受け渡しには使わない。
`load_scene_result(scene.json)`またはstore directoryを指定すると、checksum検証後にその統合結果を読める。
datasetの完成marker（`annotation.json`）は`dataset_publication.py`が公開し、
[generate_dataset](../generate_dataset/README.md)がclipごとの呼出しと結果を管理する。

モデル/設定/上流artifact/入力組立の変更は再計算を必要とする。現行のコードfingerprintは`src`全体を保守的に含むため、
コード変更は全componentを無効化する。設定・checkpointだけの変更は、そのcomponentと依存先だけを無効化する。
新しい実装を差す場合は`ComponentNode`へ実装、IO、assembler、bindings、設定を登録する。runner変更は不要。

## execute / load

`execution.<component>=execute`は同一identityの完成artifactを再利用し、無ければ実行する。
`execution.<component>=load`は採用済みartifactを必須とし、モデルを実行しない。schema/version・node名・
入力artifact参照が宣言と一致しなければ停止する。componentが生成したartifactは実行identityも一致を要求する。
`cache.source=load`は全componentの再開検証用。入力や設定が違う通常artifactを暗黙採用しない。
