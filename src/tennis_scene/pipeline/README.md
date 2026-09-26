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
`court_detection`は各cameraのframe 0だけをKP＋LINE共同推定し、`court_observations` schema v2で保存する。
`court_calibration`はこの1frameから初期校正・ROIを作り、固定cameraのコート座標を全frameへ明示的にbroadcastする。
`observed_frame_indices=[0]`と`temporal_policy`を保存し、他frameでモデルを実行したとは扱わない。

```text
動画 → court_detection → court_calibration
動画＋ROI → person_detection → person_tracking → pose_estimation
動画 → ball_detection
pose＋court → person_reid / court_side（独立モデル）
人物対応＋side＋2D観測 → camera_alignment
人物観測＋camera → player_triangulation
単一球観測＋camera → ball_triangulation → ball_smoothing
人物対応＋2D観測＋camera → body_view_selection → gvhmr
GVHMRパラメータ＋3D関節 → body_placement → scene_assembly
```

人物detectorはDINO/YOLOを選べる。trackingは保存済みbboxをBoT-SORTへ渡し、detectorを呼ばない。
ViTPoseも保存済みtrackから実観測frameを選ぶ。各cameraの累計IDは4以下、ID/slotの再利用や暗黙統合は行わない。
BoT-SORTの追跡IDが短い欠落で分裂した場合は、時間差・bbox位置と大きさ・服装色がすべて近く、候補が一意のtrackletだけを結合する。
1frameだけ重なるID交代も、重なったbboxが同じ人物を囲む包含関係にある場合だけ結合し、重複観測は古いIDのboxを採用する。
元のID、欠落/重複frame数、照合距離を`person_tracks` v3に残す。複数候補や累計4人超では明示的に停止する。
モデルの人物同一性の正しさは可視化でも検証する。

## 球の時系列平滑化

`ball_triangulation`は従来どおり各frameの2D観測から独立に3Dを計算し、`ball_smoothing`はその出力を別artifactとして受け取る。
`configs/pipeline.yaml`の`ball_smoothing.method`は既定で`none`なので、従来の3D座標がそのままsceneへ進む。
選択肢は`none`、`savgol`（11frameの局所2次多項式）、`robust_spline`（Huber重み付き加速度正則化）、
`ballistic_rts`（重力を含む状態遷移の前向きKalman・後向きRTS平滑化）の3方式。

3方式とも有効な連続区間だけを処理し、観測不足で無効になったframeは補間・外挿しない。
支持された軌道のY方向反転と、低いZ極小を切れ目として検出し、その位置は三角測量値に固定する。
これは打球／バウンドの候補であり、真のイベントラベルではない。`ball_smoothing`は元の三角測量で採用したcamera inlier maskを
出自として保持し、変更後の3D位置の再投影誤差だけを再計算する。厳密な3D正解がないclipでは、平滑さと再投影の両方を確認する。

保存済みclipで3方式を同じ原入力に適用する入口は`scripts/compare_ball_smoothing.py`。
`--scene-index <clip>/annotations/tennis_scene/scene.json --output-dir <review>`で3つのscene archive、
球軌道、比較図、`index.html`を生成する。各`scene.npz`は球3D座標のみ異なり、同じ人物SMPL mesh・
コート・有効maskを保持する。各archiveを`src.tennis_scene.scripts.visualization`の
`style.player_representation=smpl`で描画すると、比較可能なmeshフルシーン動画になる。

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
採用componentが差し替わった時点で旧exportへの公開参照を外す。既存のimmutableなexportファイルは保持し、
`scene.json`から統合sceneを読む際にも上流artifactの依存鎖が現採用版と一致するか検証する。
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
実clip検証では、学習済みRe-IDの推論artifactを先に保存した後、既存の人手人物対応を別の同schema artifactとして明示importできる。
旧GVHMRのplayer軸と現track IDはbbox時系列で一意に照合する。import後もモデルのembedding・valid mask・cosine閾値はそのまま保持し、
モデルが推論したIDと確認済みIDを両方記録する。確認済みの対象2名以外は明示的に`-1`とし、元の検出・追跡成果物には残すが、
三角測量・GVHMR・sceneのplayer軸には含めない。確認済みIDをモデルの予測精度とは扱わない。

datasetへの公開は、既存の`annotation.json`完成markerを維持しつつ、`scene_result`でimmutableなexportを指定する。
component storeをdirectoryごと置換しない。SLCS・review・residual readerも対応する公開参照を読む。

実clip qualificationの入口は`tests/benchmarks/component_pipeline.py`。GPU実行は共有training queueを使う。
外部ball・確認済みside・確認済み人物対応をloadし、他の処理を動画から実行してscene exportとload-only再開を検証する。

保存済みcomponent出力の目視確認は、repo rootから
`.venv/bin/python -m scripts.visualize_component_store --clip <clip directory> --output <review directory>`
で行う。出力先の`index.html`にcomponent別の画像・timeline・診断値が並び、`manifest.json`に使用artifact IDを記録する。
`scene.json`に未生成のcomponentは未生成と表示する。新しいartifactが増えたら同じコマンドで再生成する。
`--videos`を付けると、ball・人物検出・tracking・poseの全frame overlay動画もH.264で生成する（`ffmpeg`が必要）。
