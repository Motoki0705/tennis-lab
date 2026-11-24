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
| `batching.py` | componentが宣言した推論単位の互換分割、有限queue、作業量の上限と書き戻し |
| `inference_runtime.py` | 内容ベースの共有キー、モデルの遅延読込・借用・最終需要後の解放、計測hook |
| `model_resources.py` | 複数componentが使うViTPose・外観encoderの共通モデル宣言 |
| `clip_inputs.py` | datasetのclip選択・manifest/media解決、sourceの同期・frame数検証、実行前後の入力照合 |
| `dataset_publication.py` | dataset完成markerの再利用判定、scene契約検証、`annotation.json`の原子的公開と失敗marker |
| `storage/codec.py` | componentが宣言したOutput型をJSONと数値配列へ保存・復元。pickleや動的class importは使わない |
| `storage/clip_store.py` | clipを正本とする永続store、原子的な成果物公開、`scene.json`、メモリcache |
| `storage/scene_export.py` | 統合sceneのimmutableなexportと`scene.json`への採用 |
| `orchestrator.py` | 共通入力部品への動画束縛、標準recipeから1clipの計画を作成・実行・最終化、実行receipt |
| `video_batch.py` | 同じvideoの固定clip群を1計画へ束縛し、失敗を分離してclip完成直後に公開する入口 |
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
`ComponentNode.after_components`はnode名の`/`より前で指定した、選択されたcomponent群の間に実行順の制約を置く。
全clip/cameraの対象群が完了するまで後段をreadyにせず、部分cache/loadでもモデルのpassを揃える。
この制約はartifactの入力依存・cache identityへ加えず、target閉包にも含めない。
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
失敗・未処理のclipは最終化しない。モデルの共有と寿命はruntimeが管理し、runnerはready nodeのbatchを実行する。
既存の`ComponentRunner(nodes, store)`・`run(targets=...)`・`output(name)`は1clipの同じ計画へ委譲する。

単一動画入口は`prepare_clip_source`で実mediaの時間軸を検証し、入力hashを実行後・export前に照合する。
dataset入口は`select_dataset_clips`・`resolve_dataset_clip`で選択・manifestとrecordの時間軸・mediaを検証し、
同じ単一clip入口を呼ぶ。完成markerの呼出し順序と失敗継続は
[generate_dataset](../generate_dataset/README.md)を参照する。共通関数は複数clip入口からも個別に呼び出せる。

## 経路の選択

`recipe` は `definition.py` のノード・依存bindingを選ぶ。省略時と `recipe=standard` は従来の宣言を使う。component override は選択した経路にあるノードへ適用する。

別経路では、component store、dataset publication、単体実行の出力がそれぞれ従来の保存先の `recipes/<recipe>` に入る。モデル・実装・前処理の設定は該当ノードの成果の識別に使われ、下流は依存成果のdigestを参照する。

`recipe=fasthmr` の入力、必要な資産、実行可能条件は [FastHMR](FASTHMR.md) を参照。

## 横断バッチの設定

`run_video_batch`の`batching.enabled=true`で、推論componentが`batch_adapter()`を宣言したstageに横断バッチを使う。
adapterはモデルalias、推論単位の列挙、互換キー、forward、書き戻しを持つ。
推論単位の宛先はclip・camera・source frameと、該当する検出row、Ballのwindow、HMR2の人物・segmentで決まる。
同じモデル宣言を持つready nodeを複数clipから優先して開き、sourceごとに順次decodeする。
互換キーごとに有限queueを設け、満杯になったbatchと末尾を処理する。完成したcamera/clipの成果物を順次公開する。

| キー | 初期値 | 単位と制約 |
|---|---:|---|
| `batching.sizes.court` | 2 | 各cameraのframe 0。推論入力のパディング後shapeが同じ画像を組み合わせる。領域探索の候補評価は各画像の既存順で実行 |
| `batching.sizes.dino` | 2 | resize後の形状が同じ全画面入力。元の画像サイズでboxを復元 |
| `batching.sizes.vitpose` | 4 | 検出crop数。flip test有効時の実forwardは最大8行 |
| `batching.sizes.clip` | 16 | 有効な外観crop。元の検出rowにembeddingを書き戻す |
| `batching.sizes.hmr2` | 4 | 身体view選択済みcrop。cameraごとに連続decodeし、人物・segment・frameへ書き戻す |
| `batching.sizes.ball` | 1 | 固定長窓のB軸。T軸は元の1clip・1camera内で、backfill、overlap所有窓、gateと平滑化を維持 |
| `batching.max_sources` | 4 | 同時に開く推論source数。通常はcameraごと、HMR2はclipごとに数える |
| `batching.max_pending_items` | 32 | 互換キー全体でqueueに保持する推論単位数。全モデルのbatch上限以上 |
| `batching.max_working_bytes` | 3 GiB | decode作業領域、queue内の入力、前後処理、保持中のfeature/evidence、読み出すartifactとSMPL頂点を含む明示的な作業量 |
| `batching.store_cache_bytes` | 0 | 全clipのClipStoreが共有するLRUのバイト上限。0でディスクから読む |
| `batching.clips_per_pass` | null | 既定では選択した全clipを1回のruntimeで実行。整数で1passあたりのclip数を指定 |

作業量の上限を超える場合は、その時点の要求量を示して停止する。ViTPoseはflipで増えるcropとheatmap、
Ballは固定長のrolling window、stack、heatmap、候補patch、camera単位の蓄積配列を見積もる。
X3DではFP32のRGB・MDDと結合後の5チャネル入力の同時保持分を、forward前に窓ごとに予約する。
720×1280・T32でこの前処理は1窓あたり1,125 MiB。固定3clipの最長356 frameを4 sourceで
同時に処理すると、B=1の予約は約2.24 GiBに収まる。B=2では約3.69 GiBとなり、上限の明示指定が必要。
Body placementはclip全体のSMPL頂点と人物ごとの一時配列を考慮する。
作業量は配列の予約ベースで、モデルの重み・内部activation、OpenCV内部のbuffer、OSのfile cacheを含む
process全体のRSS/VRAMはrecorderの実測値で確認する。CUDAエラー・OOMは呼出し元へ伝播する。
batch上限・精度・対象を変更する再実行は設定で指定する。
ClipStoreはcache hitでも配列のchecksumを検証し、破損したentryを破棄してエラーを返す。

同じadapterのB=1条件は、`batching.enabled=true`に加えて
`'batching.sizes={court:1,dino:1,vitpose:1,clip:1,hmr2:1,ball:1}'`を指定する。
`batching.enabled=false`は各componentの既存バッチを使うモデル再利用条件となる。
比較時は同じ入力・重み・精度と独立したstoreを使う。GVHMRは選択済みsegment単位で実行する。
初期値でのGPU速度、充填率、品質差の受入記録は[固定3clip比較 #1757](https://github.com/Motoki0705/tennis-lab/issues/1757)で扱う。

## モデルの所有と寿命

推論を持つcomponentは`InferenceComponent`を実装し、`model_specs()`でaliasと`ModelSpec`を宣言する。
`process_with_models(inputs, models)`は`models.borrow(alias)`で借りる。宣言やcache判定ではモデルを起動しない。
通常の`process(inputs)`も、同じruntimeを1nodeの寿命で使う。componentには重みを保持しない。

`ModelSpec`はモデル種別・architecture/encoder・device・dtype・前処理/推論設定と、名前付き資産path、
生成・初期化・解放のcallbackを持つ。`key`は資産のSHA-256と設定のcanonical JSONを含み、
clip/camera IDと資産pathを含めない。同じ内容の別pathは共有でき、異なる重み・設定は共有しない。
checkpoint内のarchitectureは重み全体の内容identityにも含まれる。
手動の`ComponentNode`も、`settings.models`へ`model_identities(component.model_specs(), catalog.file_identity)`で
モデル宣言を束縛する。execute/cache判定前に実際の宣言と照合し、未束縛・差し替えを拒否する。
この照合は束縛済みdigestを使い、checkpointを読み直さない。明示loadは保存済みの契約・identityを照合する。
`standard_definition(overrides=...)`とorchestratorの`components`は、選択したcomponentの実ModelSpecを
nodeと完成markerの`component_overrides`へ含める。モデル宣言がない差し替えは、処理設定を返す
`cache_identity() -> Mapping`が必要となる。モデル以外の追加設定もこのmethodで宣言する。
クラス名だけの差し替え・未知のnode名は拒否し、標準componentへ置き換えない。
runtimeはnodeの成果物settingsに`file_identity`で束縛したSHAを`expected_assets`として受け取り、
初回借用時もそのSHAと照合する。宣言済み資産の置換・欠落は公開前に停止する。
開始した計画内でキーを固定し、借用時・初期化直後の資産変更も停止する。
Courtはcheckpoint metadataをCPUで読み、同じresolverで解決した外部DINOv3 backboneとrepository内の
全Python source（相対名と内容SHA）を共有キーと成果物settingsの`external_assets`に含める。
通常clipの完成markerを照合するpublication identityにも同じCourt資産を含め、外部資産の変更を再利用で見逃さない。
cache/loadでも資産の宣言・metadata検証は行うが、推論モデルは構築しない。
同じ`CourtKPConfig`で構築するnode/clipは`asset_catalog`でmetadataと内容digestを共有する。
checkpointのCPU metadata読込と全量hashは同じ資産につき各1回。statの変更でcacheを無効にし、
metadata読込中の変更も拒否する。新しいconfig（`dataclasses.replace`を含む）は新しいcatalogを作る。
全推論nodeの成果物settingsと完成markerは、標準のcomponent構築と同じ`ModelSpec`から有効な
device・dtype・architecture・推論設定・資産を列挙する。標準poseはtrackingのevidenceとその依存identityを使う。
Court成果物の来歴は各nodeの束縛から記録する。
人物検出は外部DINOのPython sourceと、指定されている場合はDINO native extensionの内容も含める。
未指定のextensionは実ローダーが明示的に停止し、宣言後のextension選択変更も初期化前に拒否する。
外観encoderのfactoryには宣言・SHA照合に使ったcheckpoint pathをそのまま渡す。
外観DINOv3も外部Python実装の内容をキーに含める。実hub loaderはimport済み`dinov3` namespaceの
実装identityをprocess内で固定し、安全に切り替えられない別実装や未管理の事前importは明示拒否する。
実装を切り替えるときは新しいprocessを使う。同内容の別pathは共有できる。
hubからimportした`dinov3.*`の`__file__`も宣言inventoryと照合する。外部package・外部子module・出自不明を拒否し、
失敗したimportを同じprocessで再試行しない。既存namespaceを別repositoryから再loadする場合も、読込元が異なれば
新しいprocessが必要となる。同内容のモデルをruntime内で共有する内容キーの契約は変わらない。
人物検出DINOも、models/utilのimportとbuild後に同じ出自検査を行う。
repository配下のsymlink（ディレクトリ・循環・切れたリンクを含む）は、Python import対象の全列挙を保証できないため生成前に拒否する。
dataset入口は完成marker identityの資産スナップショットを保持し、計画の各nodeが同じ内容を
束縛していることを実行前に照合する。identity取得から計画作成までのcheckpoint置換も停止する。
単一入口と一括入口は共通の最終化で、入力・共有court（driftを含む）・共有court_side・資産を
export直前に再検証する。既存exportは記録済みchecksumと再構成結果を照合してから採用し、
同じ世代の作成・検証・index更新をclip lockで直列化する。
既知の制限: 実行中にPython sourceを追加・書換する運用は対象外。実行中の資産は固定する。
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
計測hookの`__exit__`が例外を抑制しても、factory・initialize・release・cleanupの失敗は外側へ伝播する。
モデル操作とhookの双方が失敗した場合は例外groupへ保持する。
例外の構造とstack位置は診断用に保持する。成果物の`store.publish`・active参照更新・成功statusは
nodeの解放・hook完了後に行う。解放失敗では旧active参照を保持し、新しいexecutorでも失敗出力をcache hitしない。

共有対象はCourt、DINO、Ball、ViTPose、CLIP（追跡・選別の画像推論経路・人物対応）、HMR2、GVHMR、
身体配置のSMPL-X/変換行列/関節・root regressor。tracker・AFLink・時系列入力は共有モデルへ混ぜない。
人物特徴はViTPoseとCLIPの別nodeで推論し、追跡と標準選別・pose出力は保存特徴だけをCPUで読む。
HMR2画像特徴とGVHMRも別nodeで処理し、HMR2の最終需要の後に解放してからGVHMRの時系列推論へ進める。
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

`court_detection`・`person_detection`・`person_pose_features`・`person_appearance_features`・`person_tracking`・`player_selection`・`pose_estimation`・`ball_detection`はcameraごとに独立する。
既定の `court_kp.mode=clip` では、`court_detection`は各cameraのframe 0だけをKP＋LINE共同推定し、`court_observations` schema v2で保存する。
明示的な `video_shared` は[動画共有コート成果物](video_court/README.ja.md)をbuild/loadし、同じvideo・cameraの共通Hとdrift診断をschema v3で入力する。
`court_calibration`はこの1frameから初期校正・ROIを作り、固定cameraのコート座標を全frameへ明示的にbroadcastする。
`observed_frame_indices=[0]`と`temporal_policy`を保存し、他frameでモデルを実行したとは扱わない。
人物検出は校正成否に依存せず全画面で実行する。未校正cameraの選手選別は理由付きの空結果を保存する。

```text
動画 → court_detection → court_calibration
動画 → person_detection → person_pose_features → person_appearance_features
検出＋保存pose・appearance → person_tracking
track＋court → player_selection → pose_estimation
動画 → ball_detection → ball_points
ball_points＋court または player_selection＋court → court_side
選別group＋court＋side（＋動画のcrop） → player_association
人物対応＋side＋2D観測 → camera_alignment
人物観測＋camera → player_triangulation
ball_points＋camera → ball_triangulation
人物対応＋2D観測＋camera → body_view_selection → hmr2_features → gvhmr
GVHMRパラメータ＋3D関節 → body_placement → scene_assembly
```

人物検出はDINOに固定し、標準設定は**COCO DINOの全画面、score ≥ 0.30、入力800/1333**（#964の2026-09-30判断）。
ROI gateを置かず、選手以外も2D候補として保存する。#937は
`people_models.dino_checkpoint=player_detection/chat-player-v1-e8-best-pr937.pth`を明示した場合に使用できる。
指定重みが無い・DINO形式でない場合は停止し、別の重みを選び直さない。
`person_detections` v2のcheckpoint hash・全画面scopeがartifact identityと下流の依存参照を変える。
旧ROI検出・pose・文脈を新しい経路の成果物として再利用できない。

### 人物特徴の保存とCPU追跡

単一clip入口と一括入口は同じ`standard_definition`から、cameraごとに次のnodeを作る。

| node / schema | version | 数値配列と契約 |
|---|---|---|
| `person_pose_features/<camera>` / `person_pose_features` | 2 | `frame_offsets` int64 (T+1)、`source_rows` int64 (N)、`keypoints` float32 (N,17,3)、`model_identity`・`preprocessing_identity`。元画素x/yと加工しないheatmap peak |
| `person_appearance_features/<camera>` / `person_appearance_features` | 4 | 同じoffset/row、`embeddings` float32 (N,1280)、`appearance_valid` bool (N)、CLIP encoder名・`encoder_identity`・`preprocessing_identity`。再利用が有効なら検出のあるframeのデコード済みBGR画像のSHA-256を`frame_sha256` uint8 (T,32)に記録する。無効cropはzero＋false |
| `person_tracking/<camera>` / `person_tracks` | 7 | 検出v2・pose特徴v2・外観特徴v4のcamera・全frame offset・元row・期待するモデル/前処理identityを照合し、CPUでStrongSORT++/AFLink/GSIを実行 |
| `player_selection/<camera>` / `selected_player_tracks` | 3 | 保存した追跡特徴の選別とgroup軸への対応付け |
| `pose_estimation/<camera>` / `person_poses` | 2 | 選別groupの実観測poseを再利用。標準経路で再推論しない |

appearanceの入力は検出だけで、標準CLIPへpose promptは渡さない。外観だけのtargetはViTPoseを要求しない。
`after_components`で、選択された全camera/clipの人物検出→pose→CLIPを順に処理する。
ViTPoseとCLIPはruntimeの別passで借り、それぞれ最終需要後に解放する。部分cacheからの再開でも順序を保つ。
両特徴の`cache_identity()`は各moduleが使う前処理だけを含み、override時と完成markerにも反映する。
load/importの採用時は、宣言したModelSpec（重みの内容・精度・設定）と前処理を保存metadataへ照合する。
CPU追跡も同じ照合をAFLink/StrongSORTの開始前に行い、不一致や不明な入力は拒否する。
`execution.person_pose_features`・`execution.person_appearance_features`は他nodeと同じexecute/load契約。
追跡・選別・poseの旧schemaは新しいidentityとして読み替えず、executeで再生成し、loadは不一致で停止する。
特徴のsource・重み・前処理・batch上限・dtype・実装identityと上流artifact参照は既存storeに保持する。

ViTPoseは従来のsquare box/enlarge、半解像度のcrop、設定済みflip_testを保持し、設定されたforward精度を共通ModelSpecから受け取る。保存するkeypointとconfidenceは常にfloat32。
複数frameのcropを`people_models.runtime.vitpose.batch_size`まで集約する。
appearanceは従来のrounded/clipped box、RGB crop、画像対角に応じた最低高さを保持し、
`FeatureConfig.appearance_batch_size`（既定64）まで複数frameを集約する。
`features.py`のcrop準備・batch推論・元rowへのscatterは共通で、既存の1frame `FeatureExtractor`も使う。
動画は各passで順次decodeし、RGB/crop全体を保持しない。空frame・端数batch・空の末尾も保ち、
各cameraの数値特徴をnode完了時に直ちに原子的に保存する。clip/cameraをまたぐcrop batchは扱わない。

`backend=pytorch` の `people_models.runtime.dino_detector.precision` と `people_models.runtime.vitpose.precision` に
`fp32`（既定）、`bf16`、`fp16` を個別に指定できる。半精度はモデルforwardのautocastで選び、
ViTPoseのheatmap復号とDINOのscore・box復号はFP32で行う。DINOの変形可能アテンションは
CUDA拡張の入力dtypeに合わせて、混合精度条件でもブロック全体をFP32で実行する。
CUDAのBF16にはネイティブ対応GPUが必要で、ViTPoseのBF16はCPUでも利用できる。
BF16・FP16と `people_models.runtime.vitpose.flip_test` は各段のartifact identityに反映される。
既定と明示のFP32は同じ設定identityを使う。
`fp32_cache_compatibility.json` に記録したソース差分の全ファイルが一致する場合は、その差分を
旧版のハッシュで表す。C3のモデルruntimeと追跡schemaの変更を含む現行構成では、この精度変更だけの
互換条件は成立しない。旧schemaの成果物は再生成する。ほかのソース、入力、重み、設定、上流参照の変更もidentityに反映される。

標準追跡は`person_tracking.method=strongsort_pp_pose`だけを受け付ける。StrongSORT++＋pose/CLIPのrun 10固定profileを
productionと#935用の[共通入口](../../tasks/person_tracking/README.md)で処理する。
`person_tracks` v7は元検出box/row・pose・CLIPを持ち、AFLinkで結んだ元IDも保存する。
CLIP evidenceには`ModelBorrower.identity()`で得た内容SHA・設定を含むencoder identityも保存し、regroupで保持する。
GSIの`reconstruction.boxes/interpolated`は別配列で、`observed`は実検出のみ。
選別・人物対応・pose出力はsynthetic boxを観測へ入れない。旧方式はtaskの比較benchmarkで使用する。
重み・方式・特徴契約のエラーは停止し、別方式へ戻さない。
追跡の進捗は`person_tracking/<camera ID>`ごとに、空検出を含む処理済みframe数と総数を表示する。
端末では1本のbar、リダイレクト時は開始・最大30秒に1回の途中経過・最後の実数を出す。
この数は保存特徴のframe単位のCPU処理を表し、その後のAFLink/GSI処理の完了を意味しない。
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
`selected_player_tracks` v3の`selected`は元track軸の全実観測を保持し、領域外・無効足元を削らない。
追跡特徴は選別側のencoder名とモデルidentityが一致するときだけ再利用する。不一致・identity不明なら
設定した選別encoderで明示的に再計算し、appearance診断へsourceと理由を記録する。
旧person_tracks v5/v6・pose特徴v1・外観特徴v1〜v3・selected_player_tracks v2はexecuteで再生成する。galleryでは旧形式も表示できる。
元boxは参照先`person_tracks`に保存され、`raw_track_ids`で対応する。
poseと既存v3人物対応へ渡す`tracks`は1 group/frameの時系列で、handoff重複だけを小さい元ID優先でまとめる。
`origin_rows`と連結診断に出自を保存する。group IDとraw tracker IDを混同しない。
追跡前のViTPose/CLIPを元rowでgroup軸へ移し替える。欠落を実観測として補間しない。

`player_association`（`components/identity.py`、`person_identities` schema version 3）は
[src/tasks/player_association](../../tasks/player_association/README.md)の対応付けを、校正済みcameraのtrackのboxと
`court_side`が決めたsideに適用する。方式のパラメータはtaskの採用済みdefault YAMLを読み、
シングルス/ダブルスは`player_association.players_per_side`で明示する。CLIPの重みを資産identityに含める。
`player_association.reuse_clip_embeddings`は既定の`false`で全sampleを再推論し、`true`で保存特徴の再利用を有効にする。
`true`では通常のsampling後に元検出row・box・画像・encoder・前処理を照合し、一致した個別embeddingを読む。
`true`に切り替えた初回は画像SHA-256を含む外観特徴を生成する。
照合できないsampleはカメラ内で最大64cropずつ同じCLIP runtimeで推論する。方針はcache identityに含まれ、
`diagnostics.appearance`に再利用数・再推論数とtrackごとのframe・理由が残る。
C9を有効にした場合は、実際に選ばれた動画デコードbackend・captureの設定とcropの環境を、外観特徴・人物対応の
cacheおよびclipの完成markerに記録する。推論時に開き直した動画も照合する。条件の異なる成果物はexecuteで再生成し、
loadは不一致で停止する。古い完成markerから再生成するときは`overwrite=true`を指定する。
比較用の旧尺度・別encoderはtaskの比較APIから使用する。
player IDはframeごとで（`player_ids` (V, D, T)）、1本のtrackがID switchの前後で別の人物を持てる。
決まらないclipは`ReconstructionUnavailable`（reason `player_association_<停止理由>`、全scoreと途中の決定）で停止する。

`court_side`は[src/tasks/court_side](../../tasks/court_side/README.md)の仮説検定を、校正済みcameraと選択した証拠の
~30fps格子に適用する（schema version 3）。成果物は採用したhalf-turn、全仮説のcost/support/使用frame数、marginを持つ。
既定のclipモードで決まらない場合は`ReconstructionUnavailable`（reason `court_side_<停止理由>`、仮説ごとの証跡）で停止する。
固定カメラでは[動画共有のcourt side](video_court_side/README.ja.md)を選べる。全clipの有効frameで一度判定し、単独判定が矛盾するclipを失敗、単独未決を既定でwarningにする。
選手だけの再構成は `court_side.evidence=player` と [ボール関連段の無効化](../../tasks/court_side/README.md#プレーヤーから判定する) を指定する。
`camera_alignment` は決まったsideを1仮説として、計画内の人物・ball観測で絶対的な整合性を再検証する。

`body_view_selection`は人物ごとに観測frame数、平均信頼度、camera ID順で1viewを選び、実行区間を成果物化する。
`hmr2_features`は選択済みの人物cropからHMR2画像特徴を生成する（`hmr2_features` schema v1）。
成果物はclip ID、人物ID、camera ID、source動画のSHA-256、選択順のsegment番号、
segment内のsource frame番号とfloat32 `(L,1024)`特徴を保持する。sourceの動画path・サイズ・frame範囲と
選択結果を推論前に照合し、人物やsegmentをまたいで時系列を連結しない。
crop前処理・batch上限・重みは従来と同じHMR2 wrapperを使い、clipをまたぐbatch化は行わない。

`gvhmr`は`body_view_selection`と保存済み`hmr2_features`を受け取り、clip・人物・camera・source動画・
segment・frameの対応をモデル起動前に検証する。特徴の欠損・不一致を画像からの再推論で補わない。
既存のsegment単位で時系列推論し、`body_parameters` schema v2を保存する。
`body_placement`もv2を要求する。旧v1はexecuteで再生成し、loadではschema不一致として停止する。
SMPLのmesh/COCO17変換と位置・yaw・scaleの配置は`body_placement`が担当する。
`execution.hmr2_features=execute|load`は他nodeと同じcache・依存照合に従い、
単一clipと複数clip計画は共通の`standard_definition`・assembler・runtimeを使う。
HMR2・GVHMR・body geometry資産は各段階の最終需要まで共有し、人物ごとには読み直さない。
`gvhmr`の段階barrierは、要求対象の全clipの`hmr2_features`完了後に時系列推論を開始する。
保存特徴のあるclipと未計算のclipが混在する再開でも、HMR2の解放をGVHMRの読込より先に行う。
この順序制約はclipをまたぐartifact依存には含めず、部分targetでは選択していないclipを実行しない。
両推論nodeのModelSpecは共通のcomponent生成表から成果物settingsと完成markerへ束縛する。
HMR2特徴とGVHMRは既定でFP32を使い、DINO・ViTPoseとは別に推論バックエンドを指定する。
身体配置はFP32を使う。
上流の精度変更は依存artifactのidentityを通じて身体段のcacheへ反映する。

## TensorRT FP16

ViTPose-H、DINO Swin-L、CLIP-ReID ViT-B/16、HMR2、GVHMR SIGA24を個別に選択できます。
既定のバックエンドはPyTorch FP32です。TensorRTの任意依存はLinux x86-64とCUDA 13用です。

```bash
uv sync --locked --extra tensorrt
```

各設定は `people_models.runtime` にあります。TensorRTを使うモデルに
`backend=tensorrt`、`precision=fp16`、`engines`、`batch_size` を指定します。
`engines` は `paths.artifact_root`（既定 `outputs`）相対のディレクトリ一覧です。

| 設定キー | `batch_size` の単位 | エンジンの入力 |
|---|---|---|
| `vitpose` | pose crop | 256×192。flip有効時の入力はcrop数の2倍 |
| `dino_detector` | 同じresize後shapeのフレーム | H×Wごとに1ディレクトリ。MSDAは標準GridSample |
| `clip_reid` | RGBの人物crop | 256×128。追跡特徴と人物対応のCLIP-ReIDで共用 |
| `hmr2` | 身体viewのcrop | 256×256 |
| `gvhmr` | 同じ長さの人物系列 | 人数・系列長はprofileの範囲。復号・静止カメラ補正・IKはFP32 |

TensorRTを選んだモデルの `batch_size` は単一clipと横断バッチで共通です。
横断バッチを有効にしたとき、PyTorchのモデルには既存の `batching.sizes` が適用されます。
各バッチの端数を処理します。GVHMRは連続する同じ長さの人物系列をまとめ、人物ごとに復号します。
5モデルを同時に使った評価時の `batch_size`、必要なprofile、メモリ実測は[全クリップ評価](../../../knowledge/runs/run-i2363-trt-player-video/README.md#バッチサイズの根拠)に記録しています。

### DINO GridSampleの単体精度

DINOのTensorRT FP16は明示設定で選択でき、既定はPyTorch FP32です。
固定64画像の実測ではIoU P05が0.93056、score差P95が0.01724で、
単体基準のIoU P05 ≥ 0.98、score差P95 ≤ 0.01をともに満たしていません。
常設化前のPhase 1でもIoU P05は0.93252、score差P95は0.01726で、同じ2項目が基準外でした。
採用は全クリップE2Eで判断します。
[実測記録](../../../knowledge/nodes/tennis_scene/000081-run-i2362-tensorrt-pipeline.md)・
[全クリップE2E #2363](https://github.com/Motoki0705/tennis-lab/issues/2363)

### エンジンの作成と配置

保存規約は `outputs/tensorrt/<model>/<shape>/<build-tag>/` です。
`<model>` は `vitpose`、`dino`、`clip`、`hmr2`、`gvhmr` のいずれかです。
`<shape>` は前処理後のH×W、GVHMRでは系列長のmin・opt・maxを表します。
ビルドごとに新しいタグを使います。

```bash
bundle=outputs/tensorrt/vitpose/256x192/example-b2
.venv/bin/python -m src.tennis_scene.scripts.tensorrt export vitpose "$bundle"
.venv/bin/python -m src.tennis_scene.scripts.tensorrt build "$bundle" --opt-batch 2 --max-batch 2
```

保存先はworktreeルートからの相対パス、または絶対パスで指定します。
ローカルの変換・ビルドは[共有training queue](../../../.agents/skills/training-queue/SKILL.md)で実行します。
exportはcheckpointやViTPose headの設定を `pipeline.yaml` から読みます。
個別設定は `--override KEY=VALUE` で渡し、DINOにはresize後の `--image-size HEIGHT WIDTH` を指定します。
GVHMRのbuildには `--frames MIN OPT MAX` が必要です。長さのprofileに加えて
`--opt-batch` と `--max-batch` で人数を指定します。
新しいONNX exportはattention maskを `scores + Where(mask, -32768, 0)` に書き換えます。
加算mask導入前のB=1 planは、元の設定とprofileを照合したうえでB=1の実行に使えます。
新しいplanを検証する場合は `export gvhmr` に `--keep-mask-reference` を追加し、
build後に次を実行します。ONNX Runtimeで書換え前後の全出力をバイト比較し、
TensorRTでprofile境界とL=177（範囲内の場合）の有限性・層精度を調べます。
`--progress` で実行receiptに含まれる全shapeを追加できます。

```bash
.venv/bin/python -m src.tennis_scene.scripts.qualify_gvhmr_tensorrt "$bundle" outputs/gvhmr-qualification.json
```

画像モデルの最小バッチは1です。profile内のバッチサイズは設定だけで変更できます。

```yaml
people_models:
  runtime:
    vitpose:
      backend: tensorrt
      precision: fp16
      batch_size: 2
      engines:
      - tensorrt/vitpose/256x192/example-b2
```

`export.json` は重み・外部実装、精度、`model.onnx` と外部重みのSHA-256、入出力shapeと型を固定します。
`engine.json` はそのexport、`engine.plan` のSHA-256、TensorRT版、GPU名・compute capabilityと
各入力のmin/opt/maxを固定します。実行前にこれらを照合し、ロード後にplanのbindingsとprofileを再確認します。
ファイル欠落、識別不一致、未登録shape、profile範囲外のバッチや系列長はエラーになります。
バックエンド・精度・バッチサイズ・エンジンの識別は `ModelSpec` とcomponent・publicationのidentityに含まれます。
既定のFP32では、バッチ化・計測の `batched_tensorrt_fp32_cache_compatibility.json`、初回導入の
`tensorrt_fp32_cache_compatibility.json` の順に照合します。各manifestの変更ファイルがすべて一致した場合に、
その変更を導入前のソースハッシュで表します。

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

`ball_detection → ball_points → camera_alignment / ball_triangulation` を使う。既定の `court_side.evidence=ball` でも `ball_points` を使う。
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
標準pipelineのmarker発行・再利用では、公開するsceneの依存を`body_placement`から
`body_parameters` v2、`hmr2_features` v1、共通の`body_view_selection`までたどり、
採用参照・checksumとclip/人物/segment/frameの対応を検証する。旧身体成果物は明示的に拒否する。

モデル/設定/上流artifact/入力組立の変更は再計算を必要とする。現行のコードfingerprintは`src`全体を保守的に含むため、
コード変更は全componentを無効化する。設定・checkpointだけの変更は、そのcomponentと依存先だけを無効化する。
新しい実装を差す場合は`ComponentNode`へ実装、IO、assembler、bindings、設定を登録する。runner変更は不要。

## execute / load

`execution.<component>=execute`は同一identityの完成artifactを再利用し、無ければ実行する。
`execution.<component>=load`は採用済みartifactを必須とし、モデルを実行しない。schema/version・node名・
入力artifact参照が宣言と一致しなければ停止する。componentが生成したartifactは実行identityも一致を要求する。
`cache.source=load`は全componentの再開検証用。入力や設定が違う通常artifactを暗黙採用しない。
