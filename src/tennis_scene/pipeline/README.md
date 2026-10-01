# 宣言型clip pipeline

各componentは、自身のInput/Output型と必要な成果物schemaを`ComponentIO`で宣言する。
`process(input)`は組み立て済みのInputを受け取る。runnerにモデル名・配列key・カメラ選択を実装しない。

| 場所 | 責務 |
|---|---|
| `components/` | 検出、追跡、pose、人物対応、幾何、身体復元・配置。各IO契約の所有者 |
| `input_assembly/` | 宣言済み成果物のcamera/frame/track照合とInput構築。store全体へ自由に問い合わせない |
| `definition.py` | 実装の選択、入力portとproducerの明示的な接続、モデル資産・設定の束縛 |
| `runner.py` | 宣言の互換性・循環検査、依存順実行、execute/load、公開の共通手順 |
| `storage/codec.py` | componentが宣言したOutput型をJSONと数値配列へ保存・復元。pickleや動的class importは使わない |
| `storage/clip_store.py` | clipを正本とする永続store、原子的な成果物公開、`scene.json`、メモリcache |
| `storage/scene_export.py` | 統合sceneのimmutableなexportと`scene.json`への採用 |
| `orchestrator.py` | source動画の束縛、標準recipeのcomposition、scene export、実行receipt |
| `feature_flags.py` | 無効化できる出力機能（`OPTIONAL_FEATURES`）とその依存関係 |

component名の一覧は`contracts.STANDARD_COMPONENTS`が正本で、`pipeline.yaml`の`execution`はその全てを1回ずつ持つ。

## 処理単位

`court_detection`・`person_detection`・`person_tracking`・`player_selection`・`pose_estimation`・`ball_detection`はcameraごとに独立する。
`court_detection`は各cameraのframe 0だけをKP＋LINE共同推定し、`court_observations` schema v2で保存する。
`court_calibration`はこの1frameから初期校正・ROIを作り、固定cameraのコート座標を全frameへ明示的にbroadcastする。
`observed_frame_indices=[0]`と`temporal_policy`を保存し、他frameでモデルを実行したとは扱わない。
人物検出は校正成否に依存せず全画面で実行する。未校正cameraの選手選別は理由付きの空結果を保存する。

```text
動画 → court_detection → court_calibration
動画 → person_detection → person_tracking
track＋court → player_selection → pose_estimation
動画 → ball_detection → ball_refiner_2d → ball_points
ball_points＋court → court_side（ballだけのhalf-turn仮説検定）
選別group＋court＋side（＋動画のcrop） → player_association
人物対応＋side＋2D観測 → camera_alignment
人物観測＋camera → player_triangulation
ball_points＋camera → ball_triangulation
人物対応＋2D観測＋camera → body_view_selection → gvhmr
GVHMRパラメータ＋3D関節 → body_placement → scene_assembly
```

人物検出の既定は**COCO DINOの全画面、score ≥ 0.30、入力800/1333**（#964の2026-09-30判断）。
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

`nearest_window_centre_then_earlier_start`はrefiner学習cacheと同じ、中心距離が最小の窓、
同点なら早い開始位置を採用する明示的なpolicy。このpolicyは`tail_policy=backfill`を要求し、
短clipのRGB反復を拒否する。既定の`max_score`は変更しない。

モデル実行では `evidence` は必須。注釈importと無効な検出器は `None` を明示し、
`score_semantics` で区別する。refinerは証拠なしを実検出とみなしてはならない。
下流のside・幾何・三角測量は、refiner由来の `ball_points` だけを使う。検出器の点への戻り道は無い（確率的三角測量への接続は#936）。
v1 artifactの自動補完は行わず、executeで再生成、loadはschema不一致で停止する。

## 2D ball refinerの専用recipe

`ball_refiner_recipe.ball_refiner_definition`は各cameraの
`ball_detection → ball_refiner_2d`だけを共通ComponentRunner/ClipStoreへ登録する。
標準sceneも同じrecipeを使い、各cameraのball_detection直後に全GMMを保存する。
文脈ありcheckpoint、確率的三角測量、3Dへの切替は後続の対象。
推論bundleの作成・入力契約は[task README](../../tasks/ball_refiner/README.md#推論bundleの書き出し)を参照。

recipe構築時に検出器checkpointのhashと全前処理/候補/窓設定を照合し、実際のpredictorでも
正規化と窓長を検証する。assemblerはcamera・全frame・sourceサイズ・中心距離の採用窓を照合し、
元動画をPyAVでdecodeして得たPTS/time baseを実秒へ変換する。PTSをFPSから捏造しない。
検出点の閾値・trajectory gateはrefiner入力に使わず、注釈import・証拠なしは停止する。
sourceとcheckpointのhashは実行前後にも照合する。RGBの媒体差はbundleへ明記する。

未較正`bundle` optionの出力は`ball_distribution_2d` schema v1、型は`components/ball_refiner.py:BallRefiner2DOutput`。
`prediction.distribution`に平均・Cholesky因子・混合logit・存在logitを全frame保存し、
full covariance・混合weight・存在確率は元の精度で復元できる。単位はsourceのW−1/H−1で正規化したuv。
sourceサイズ、frame/PTS/time base/実秒、detectorとrefiner両方の採用窓、未較正であることも保存する。
全欠損frameも同じ契約で保存し、点や最大成分への縮約・補間・detectorへのfallbackはしない。

以下は1cameraの実行入口。他cameraとの対応・ラベル・学習storeは要求しない。
legacyな単一点のgateやprefetch設定はsceneのpipeline.yaml、refinerが必要な入力条件はbundleが正本。
実行後に同じ引数の`--source execute`を`--source load`へ変えると、モデルを呼ばず全GMMを復元する。
code・bundle・設定・source・依存artifactの不一致や配列のchecksum不一致は停止する。

```bash
# CUDAは共有training queue経由。storeは既存の標準sceneとは分けた明示的なpathにする。
.venv/bin/python -m src.tasks.ball_refiner.scripts.run_pipeline \
  --video <絶対data-root>/clip/cam0.mp4 --camera-id cam0 \
  --ball-path e9_anchored_s42_covariance \
  --bundle <絶対checkpoint-root>/ball_refiner/i935-anchored-12k-s42 \
  --detector-checkpoint <絶対checkpoint-root>/ball_detection/i935-mixed-ft-s42-epoch09.ckpt \
  --calibration-artifact <絶対checkpoint-root>/ball_refiner/i935-anchored-12k-s42/covariance-calibration-r23.json \
  --store <絶対artifact-root>/ball_refiner/<run-id>/cam0 \
  --device cuda --source execute --detector-batch-size 4 --refiner-batch-size 32
```

### ボール経路の既定と明示option

上のCLIは採用済みの較正経路を明示する。未較正bundleの比較は
`--ball-path bundle`を指定し、較正artifactを渡さず対応するbundle/検出器を明示する。
資産の固定SHA256は[名前付きoption](../../tasks/ball_refiner/pipeline_options.py)が正本。
検出器e9、anchored_12k seed42のepoch41（checkpoint `985308b0…`）から書き出したbundle、
共分散倍率artifact `197f9e64…`（Σに1.8125148752倍）を要求する。
欠落・hash不一致・別checkpoint用の倍率はexecute/loadとも定義構築時に停止する。
manifestは重み・全入力設定・元checkpointを束縛し、倍率artifactは実行前後にも照合する。

このoptionの成果物は`ball_distribution_2d` **schema v2**、
`CalibratedBallRefiner2DOutput`。全成分のCholeskyに倍率の平方根だけを掛け、
平均・混合logit・存在logit・frame/PTS/採用窓は維持する。
倍率、元checkpoint SHA256、artifact SHA256を保存し、load-onlyでは再補正しない。
既存`bundle` optionは未較正schema v1を維持し、未知optionや暗黙の倍率1は許可しない。
標準sceneの既定は `ball_path=e9_anchored_s42_covariance`。旧ft-e13＋旧refinerは
`ball_path=comparison/ft_e13` で明示する。checkpoint-root内の配布名は `configs/ball_path/` が正本。
採用済みの実体は`ckpt/`に置き、学習runやexport元へのsymlinkで代用しない。
資産が無ければ停止し、旧重みを自動選択しない。動画は直接decodeし、JPEG化・再学習・倍率再fitはしない。
元動画3cameraのexecute/fresh-loadは通過したが、
[固定Bゲート](../../../knowledge/nodes/ball_refiner/000025-run-i935-source-b-gate-r26-20261001.md)の
GT位置誤差p90が不合格だったため、run26時点では既定を維持した。
[2026-10-01のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5921216642)
に従ってe9を既定化した。confidence規則とpoint consumer配線は実装済み。安全benchと全scene qualificationで確認する。
seedの事前判定FAILと、
それを保持して再現は十分と扱う追加ユーザー判断も同記録から辿れる。

## 信頼度を共有する点consumer

`ball_points` は保存済み全GMMを最大weight成分の平均へ縮約し、
[固定confidence規則](../../tasks/ball_refiner/README.md#点consumerの信頼度規則)で選別する。
`BallPointsOutput` / schema `ball_points` v1に点、存在確率、source px²面積、採否、規則、棄却bitを保存する。
bit1は存在確率不足、bit2は面積超過、0は採用。棄却frameは座標・confidence=0とobserved=false。
codec再読込時も規則とmaskの一致を検証する。補間・検出点へのfallbackは行わない。

`court_side`・`camera_alignment`・`ball_triangulation` は同じcameraごとのball_points artifactに依存し、
検出器のscore閾値を重ねて適用しない。court_sideのball-only/margin .15は不変。
旧ball_detectionsをpoint consumerへload/importすることはschema不一致として拒否する。
全GMMは別のartifactとして保持され、#936はそこから分布を利用できる。
ball_detectionを無効化する場合はball/refiner/points chain全体のloadを明示する必要がある。

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
datasetの完成marker（`annotation.json`）は[generate_dataset](../generate_dataset/README.md)が所有する。

モデル/設定/上流artifact/入力組立の変更は再計算を必要とする。現行のコードfingerprintは`src`全体を保守的に含むため、
コード変更は全componentを無効化する。設定・checkpointだけの変更は、そのcomponentと依存先だけを無効化する。
新しい実装を差す場合は`ComponentNode`へ実装、IO、assembler、bindings、設定を登録する。runner変更は不要。

## execute / load

`execution.<component>=execute`は同一identityの完成artifactを再利用し、無ければ実行する。
`execution.<component>=load`は採用済みartifactを必須とし、モデルを実行しない。schema/version・node名・
入力artifact参照が宣言と一致しなければ停止する。componentが生成したartifactは実行identityも一致を要求する。
`cache.source=load`は全componentの再開検証用。入力や設定が違う通常artifactを暗黙採用しない。
