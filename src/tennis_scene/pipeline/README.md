# 宣言型clip pipeline

各componentは、自身のInput/Output型と必要な成果物schemaを`ComponentIO`で宣言する。
`process(input)`は組み立て済みのInputを受け取る。runnerにモデル名・配列key・カメラ選択を実装しない。

| 場所 | 責務 |
|---|---|
| `components/` | 検出、追跡、pose、Re-ID、side、幾何、身体復元・配置。各IO契約の所有者 |
| `input_assembly/` | 宣言済み成果物のcamera/frame/track照合とInput構築。store全体へ自由に問い合わせない |
| `definition.py` | 実装の選択、入力portとproducerの明示的な接続、モデル資産・設定の束縛 |
| `runner.py` | 宣言の互換性・循環検査、依存順実行、execute/load、公開の共通手順 |
| `storage/codec.py` | componentが宣言したOutput型をJSONと数値配列へ保存・復元。pickleや動的class importは使わない |
| `storage/clip_store.py` | clipを正本とする永続store、原子的な成果物公開、`scene.json`、メモリcache |
| `orchestrator.py` | source動画とclipの束縛、標準recipeのcomposition、scene export、実行receipt |
| `imports/` | 外部成果物の明示変換。モデルのfallbackではない |
| `feature_flags.py` | 出力機能の有効/無効の整合性と旧archiveのstage名。実行DAGは保持しない |

## 処理単位

`court_detection`・`person_detection`・`person_tracking`・`pose_estimation`・`ball_detection`はcameraごとに独立する。
`court_calibration`はコート観測をまとめ、初期校正と画像内へclipした人物ROIを出す。

```text
動画 → court_detection → court_calibration
動画＋ROI → person_detection → person_tracking → pose_estimation
動画 → ball_detection
pose＋court → person_reid / court_side（独立モデル）
人物対応＋side＋2D観測 → camera_alignment
人物観測＋camera → player_triangulation
単一球観測＋camera → ball_triangulation
人物対応＋2D観測＋camera → body_view_selection → gvhmr
GVHMRパラメータ＋3D関節 → body_placement → scene_assembly
```

人物detectorはDINO/YOLOを選べる。trackingは保存済みbboxをBoT-SORTへ渡し、detectorを呼ばない。
ViTPoseも保存済みtrackから実観測frameを選ぶ。各cameraの累計IDは4以下、ID/slotの再利用や暗黙統合は行わない。
追跡IDが分裂して容量を超えた場合は明示的に停止する。モデルの人物同一性の正しさは別途検証が必要。

`body_view_selection`は人物ごとに観測frame数、平均信頼度、camera ID順で1viewを選び、実行区間を成果物化する。
`gvhmr`はHMR画像特徴とGVHMRだけを実行する。SMPLのmesh/COCO17変換と位置・yaw・scaleの配置は`body_placement`が担当する。
旧一括GVHMR chainと旧3D PLCS/BLCS componentは撤去し、歴史的手動対応のdecoderだけ`artifact_schemas/`に残す。

## 成果物

構造化clipでは`<clip>/annotations/tennis_scene/`をstoreとする。
単独動画の入口では明示的な`cache.directory/<source hash>`を使用する。

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
大配列は`.npy`へ分離する。読み込みはmmapを使い、メモリにある同じartifactを再利用する。
配列を含む保存・復元結果を下流へ渡すため、同じ実行内と再起動後で型や軸順が変わらない。

出力は一時directoryで完了してから公開し、clip単位のlockを取って`scene.json`を更新する。
中断途中の出力を下流へ渡さず、既存の完成artifactを上書きしない。
`scene.json`の`artifacts`は採用版、`exports.scene`は完成した統合出力とその入力版を指す。
`scene.npz`は派生した統合結果で、component間の受け渡しには使わない。
`load_scene_result(scene.json)`またはstore directoryを指定すると、checksum検証後にその統合結果を読める。

モデル/設定/上流artifact/入力組立の変更は再計算を必要とする。現行のコードfingerprintは`src`全体を保守的に含むため、
コード変更は全componentを無効化する。設定・checkpointだけの変更は、そのcomponentと依存先だけを無効化する。
新しい実装を差す場合は`ComponentNode`へ実装、IO、assembler、bindings、設定を登録する。runner変更は不要。

## execute / load

標準設定は各componentを`execution.<component>=execute`とする。
executeは同一identityの完成artifactを再利用し、無ければ実行する。
`execution.ball_detection=load`等は保存結果を必須とし、モデルを実行しない。
`cache.source=load`は全componentの再開検証用。入力や設定が違う通常artifactを暗黙採用しない。
明示importされた外部成果物は出力schema・source・依存先の一致を要求し、モデル重みへの依存を作らない。

外部`video_ball_annotation.v2`は元動画のSHA/FPS/サイズと0始まりの全frame対応を確認する。
観測、補間、遮蔽推定、未解決を`point_kind`で保存し、幾何の実観測には`observed`だけを用いる。
confidenceは観測1/それ以外0の採用weightで、検出確率ではない。原注釈の`image_score`を確率へ転用しない。
sideの明示importは確認根拠を記録し、モデルlogitsは生成しない。

datasetへの公開は、既存の`annotation.json`完成markerを維持しつつ、`scene_result`でimmutableなexportを指定する。
component storeをdirectoryごと置換しない。SLCS・review・residual readerも対応する公開参照を読む。

実clip qualificationの入口は`tests/benchmarks/component_pipeline.py`。GPU実行は共有training queueを使う。
外部ballと確認済みsideをloadし、他の処理を動画から実行してscene exportとload-only再開を検証する。
