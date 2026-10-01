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

`court_detection`・`person_detection`・`person_tracking`・`pose_estimation`・`ball_detection`はcameraごとに独立する。
`court_detection`は各cameraのframe 0だけをKP＋LINE共同推定し、`court_observations` schema v2で保存する。
`court_calibration`はこの1frameから初期校正・ROIを作り、固定cameraのコート座標を全frameへ明示的にbroadcastする。
`observed_frame_indices=[0]`と`temporal_policy`を保存し、他frameでモデルを実行したとは扱わない。
校正できなかったcameraはROIを持たず、そのcameraの人物検出は実行しない（ROIなしの検出はしない）。

```text
動画 → court_detection → court_calibration
動画＋ROI → person_detection → person_tracking → pose_estimation
動画 → ball_detection
ball＋court → court_side（ballだけのhalf-turn仮説検定）
track＋court＋side（＋動画のcrop） → player_association
人物対応＋side＋2D観測 → camera_alignment
人物観測＋camera → player_triangulation
単一球観測＋camera → ball_triangulation
人物対応＋2D観測＋camera → body_view_selection → gvhmr
GVHMRパラメータ＋3D関節 → body_placement → scene_assembly
```

人物detectorはDINO/YOLOを選べる。trackingは保存済みbboxをBoT-SORTへ渡し、detectorを呼ばない。
ViTPoseも保存済みtrackから実観測frameを選ぶ。各cameraの累計IDは`person_observations.max_tracks_per_camera`以下で、超えたclipは停止する。ID/slotの再利用や暗黙統合は行わない。
BoT-SORTの追跡IDが短い欠落で分裂した場合は、時間差・bbox位置と大きさ・服装色がすべて近く、候補が一意のtrackletだけを結合する
（閾値は`TrackletLinkPolicy`で、成果物identityに含む）。
1frameだけ重なるID交代も、重なったbboxが同じ人物を囲む包含関係にある場合だけ結合し、重複観測は古いIDのboxを採用する。
元のID、欠落/重複frame数、照合距離を`person_tracks` v3に残す。複数候補や累計track数の上限超では明示的に停止する。

`player_association`（`components/identity.py`、`person_identities` schema version 3）は
[src/tasks/player_association](../../tasks/player_association/README.md)の対応付けを、校正済みcameraのtrackのboxと
`court_side`が決めたsideに適用する。方式のパラメータは`player_association.config`（task側のYAML）、
シングルス/ダブルスは`player_association.players_per_side`で明示する。外観を使う設定ではRe-IDの重みを資産identityに含める。
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

モデル実行では `evidence` は必須。注釈importと無効な検出器は `None` を明示し、
`score_semantics` で区別する。refinerは証拠なしを実検出とみなしてはならない。
下流のside・幾何・三角測量は当面、既存の単一点観測を使う（refinerへの切替は#935）。
v1 artifactの自動補完は行わず、executeで再生成、loadはschema不一致で停止する。

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
