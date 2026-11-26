# tennis_scene

同期済みの固定カメラ動画から、camera-local CourtV2観測、人物・球の2D観測、三角測量、
GVHMRの身体復元を組み合わせてSceneResultを作ります。根拠不足は欠測または理由付き失敗として保存します。

## 標準経路

1. 各cameraのframe 0だけをKP＋LINE共同推定し、固定コートの初期校正を作る。
2. 全画面の人物検出にpose・外観特徴を付けて追跡し、コート座標で選手を選別する。各camera/frameの検出点と観測maskを保存する。
3. court side（`court_side`）を多視点の幾何で決め、そのsideで人物trackをcamera間で対応付ける（`player_association`）。証拠はballが既定で、`court_side.evidence=player` で選手の足元観測を使える。
4. 決まったsideを人物・ballで幾何検証し、近似カメラ校正をreference座標へ変換。
5. ボールの動きの外れ値を除去し、頑健な2次LOESSで平滑化。人物の同一ID観測と、各カメラの採用した実ボール観測を三角測量。
6. GVHMRの関節姿勢を保ち、三角測量COCO17へ位置・yawを時系列で配置。
7. 元動画の時間軸でSceneResult、品質mask、診断を保存。

既定の通常モデル経路はCourt・人物・球の手動入力を要求しません。
`ball_detection.method=gpt_annotation`では白紙注釈・自動継続・保存済みv3の再読込・固定幾何の
三角測量・任意dataset公開を一連で実行できます。[利用と復旧](pipeline/direct_annotation/README.ja.md)を参照してください。

対応範囲は同期・同FPS・同解像度の3〜5 view、各camera選別後`person_observations.max_tracks_per_camera` group、
シングルスまたはダブルス（`player_association.players_per_side`）、球は各camera/frame高々1検出です。
camera alignmentと身体viewの選択は、clip全体を約30fpsの格子で1回処理し、格子が
`frame_sampling.max_frames`を超えるclipは切り詰めずに拒否します。
reference未指定時は校正可能camera IDの辞書順先頭を選びます。IDはclip内でのみ有効です。

設定の正本は[configs/pipeline.yaml](configs/pipeline.yaml)、人物経路の既定と契約は[pipeline README](pipeline/README.md)を参照してください。
既定の配布名は配置規約であり、重みを自動取得・自動選定する処理はありません。
有効な機能が参照するcheckpointが欠けていれば、実行前のdefinition構築時に停止します。

DINO binaryの選択は[導入案内](../../docs/installation/README.ja.md#dino-cuda-extension)に従います。

```bash
# GPU実行は、このコマンドを共有training queueへ登録する。
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[match/cam0.mp4,match/cam1.mp4,match/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]'
```

動画はDATA、checkpointはCHECKPOINT、外部モデルはEXTERNAL_ASSET、sceneはOUTPUT、
component storeはARTIFACTのrootから解決します。[タスク出力パス](../tasks/OUTPUTS.md)を参照。
dataset生成は実clipから入力を束縛し、設定中のサンプル動画名には依存しません。

## モジュールと成果物

componentのIO宣言、入力組立、runner、clip store、保存形式とexecute/loadの正本は
[pipeline/README.md](pipeline/README.md)です。
storeの採用済みartifactは、推論もstoreへの書き込みもせずにHTML galleryで確認できます。

```bash
.venv/bin/python -m src.tennis_scene.scripts.visualize_component_store --store <clip>/annotations/tennis_scene --output <dir> [--videos]
```

source・採用artifactの依存・欠損区間・同frameのRGB/2D/3Dを確認する画面と、
`--serve --port 8903`での起動方法は[visualization/review/README.md](visualization/review/README.md)を参照してください。

## 登録済みvideoの一括再構成

`run_video_batch`は`dataset.json`に登録された同じ`video_id`の全clipを1つの
`ExecutionPlan`へまとめます。モデルを段階ごとに共有し、各推論はclip・cameraごとの
既存windowとbatch設定を使うのが既定です。`batching.enabled=true`で互換なframe・検出crop・
時間窓を複数clip/cameraから集めます。各上限と比較条件は
[横断バッチの設定](pipeline/README.md#横断バッチの設定)を参照してください。
既定のpipelineは`pipeline`です。
構成は単一入口と同じ[設定](../tasks/court_side/README.md#プレーヤーから判定する)を
`pipeline_overrides`で明示します。

| 用途 | `court_side.evidence` | `ball_detection.enabled` | `ball_reconstruction.enabled` |
|---|---|---|---|
| プレーヤーのみの本番 | `player` | `false` | `false` |
| ボール推論・再構成も含むbenchmark | `player` | `true` | `true` |

ボールありでX3D/CSN＋DPT T32を使う場合は`pipeline_config=pipeline_x3d_dpt`も指定します。
プレーヤーのみではball検出・2D点・三角測量のnodeとモデル読込を省き、sceneのballは無効maskになります。
省いた段への個別`execution.<段>=load`は共通設定検証で拒否します。
人物対応での保存CLIP再利用は共通の`player_association.reuse_clip_embeddings`を
同じoverridesで指定できます。既定は`false`で、benchmarkの比較時に明示して切り替えます。

meiji_4camの例です。
この動画はダブルスなので、`player_association.players_per_side=2`も明示します。
人物数の既定値は各side 1人です。
`court_kp.mode=clip`を選ぶ例です。

```bash
# GPU実行はこのコマンドを共有training queueへ登録する。
.venv/bin/python -m src.tennis_scene.scripts.run_video_batch \
  dataset_directory=tennis_multiview/processed/meiji_4cam/dataset \
  video_id=video_000 \
  batching.enabled=true \
  pipeline_config=pipeline \
  'pipeline_overrides=["court_kp.mode=clip","court_side.mode=clip","court_side.evidence=player","ball_detection.enabled=false","ball_reconstruction.enabled=false","player_association.players_per_side=2"]'
```

`court_kp.mode=video_shared`では、計画を組む前に`prepare_video_court`でvideo全体の
共有コートをbuildまたはloadし、同じ成果物を全clipへ束縛します。次は準備済みbundleを
ARTIFACT rootの`tennis_scene/court_shared/meiji_4cam`からloadする例です。
`VIDEO_COURT_ARTIFACT_DIGEST`へ準備APIが返したdigestを設定して実行します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.run_video_batch \
  dataset_directory=tennis_multiview/processed/meiji_4cam/dataset \
  video_id=video_000 batching.enabled=true pipeline_config=pipeline \
  "pipeline_overrides=[\"court_kp.mode=video_shared\",\"court_kp.video_shared.action=load\",\"court_kp.video_shared.directory=tennis_scene/court_shared/meiji_4cam\",\"court_kp.video_shared.dataset_id=meiji_4cam\",\"court_kp.video_shared.video_id=video_000\",\"court_kp.video_shared.artifact_digest=$VIDEO_COURT_ARTIFACT_DIGEST\",\"court_side.mode=clip\",\"court_side.evidence=player\",\"ball_detection.enabled=false\",\"ball_reconstruction.enabled=false\",\"player_association.players_per_side=2\"]"
```

dataset/videoのscopeと各clipのmediaを共有成果物へ照合します。build用request、診断成果物の
選択と検査不能時の方針は、共有コートの設定を使います。取得の規則は
[共有コートのREADME](pipeline/video_court/README.ja.md)を参照してください。

`court_side.mode=video_shared`では、その後に`prepare_video_court_side`で共有の
コート向きを1回buildまたはloadしてから、全clipの計画を作ります。
`court_side.video_shared.action`と`directory`、load時の`artifact_digest`を同じ
`pipeline_overrides`へ追加します。個別clipのmargin不足などは進捗の
`video_court_side.clips`へ残し、共有判定との矛盾はclip固有の失敗になります。
進捗には共有判定の証拠種別も記録します。player証拠のbuild/loadにはplayerの成果物を指定し、
ball証拠の成果物は流用しません。全clipのplayer観測が未準備の場合は`court_side.mode=clip`で
各clipの選別後に判定します。
設定とpinの契約は[共有court_sideのREADME](pipeline/video_court_side/README.ja.md)を参照してください。

meiji_4camでは、同じoverridesへ`people_models.runtime.dino_detector.precision=fp16`と
`people_models.runtime.vitpose.precision=fp16`も明示します。両方の既定値は`fp32`です。
精度を変更すると対応するcomponentと下流のcacheが更新されます。

`clip_ids=[video_000/clip_000,video_000/clip_001]`で明示subsetを指定できます。
別video・未知・重複・空の指定、manifestと実mediaのcamera対応・FPS・解像度・frame数の
不一致は、推論の開始前に拒否します。対象と展開済み設定は開始時に固定します。
実行中に登録されたclipは次回の対象になります。

clipの全nodeが完成すると、共通のexportとdataset publisherがそのclipの
`annotations/tennis_scene/annotation.json`を公開します。`continue_on_error=true`では、
`clip_too_long`などの入力・再構成失敗を記録し、そのclipの未完了nodeをblockedにして
他のclipを続けます。共有モデルの読込・解放失敗、CUDA/deviceエラー、中断は実行を停止します。
失敗が残ると終了コードは非0です。`continue_on_error=false`で最初の失敗時に停止できます。

同じコマンドの再実行では、同じ設定・入力の完成markerをskipし、markerがないclipでも
完成済みのcomponentをcacheから再利用します。古い設定のmarkerは`overwrite=true`を要求します。
componentの強制再計算には`pipeline_overrides=["cache.overwrite=true"]`を指定します。
壊れたartifactとexportは理由付きで拒否します。復旧時は破損箇所を確認し、対応する
再生成・新しいstoreを明示してください。

OUTPUT配下の`progress_directory/<実行ID>/progress.json`へ対象・設定・node状態・clipの
成功／skip／失敗／blocked、モデルload/unload、実batch shapeごとの件数を保存します。
隣の`recording.json`には[推論計測](../utils/inference_recording.md)がCPU時間、資源サンプル、
stage・node・モデル別の集計、横断バッチの充填率、モデルload・動画open回数を記録します。
`record_cuda=true`で同期付きCUDA計時、NVMLのGPU使用率、VRAMピークを取得します。
`record_cuda_synchronize=false`でCUDA計時用の同期を省き、同じ資源サンプルを取得できます。
`record_sample_interval`はサンプル間隔の秒数で、既定は0.5秒です。
間隔を短くするとCPU・NVML照会の負荷が増えます。サンプルごとの取得時間も記録します。
再開時は新しい進捗記録を作り、component storeから状態を照合します。
datasetの完成判定にはclipごとの`annotation.json`と採用中scene exportを使います。

## 座標・対応

観測の正本はpixel座標、sceneの2D座標はpixel/(width,height)です。
Court・ball検出器のpixel格子正規化（W-1/H-1）は`src.utils.geometry.keypoints`で境界変換します。
CourtKP14のcamera-local順は変えず、半回転は推論後の幾何だけへ適用します。

補間boxは実検出と区別し、observed_maskとjoint confidenceをvisibilityへ反映します。
無観測の人物にIDは割り当てません。通常モデルの球は、検出器の実観測から外れ値を除いたmaskで三角測量します。
診断用の補間点は観測に数えません。GPT方式の観測・推定の利用は[注釈import契約](pipeline/imports/README.ja.md)に従います。
両方式とも2view未満の時刻は3Dでも欠損です。

sideはball観測だけで決め、最低evidence、referenceとの接続性、絶対的な幾何品質、次点とのmarginを要求します。
決まらないclipは理由付きで停止します。`camera_alignment`は人物観測も加えて絶対的な幾何品質を再検証します。校正はframe 0で採用されたHomographyの投影点に対する
単一平面pinhole・無歪みの近似です。

## SceneResult v2

[schema.py](schema.py)と[archive.py](archive.py)がスキーマ・I/Oを所有します。

`metadata.scene_schema_version=2` のsceneは、再構成の有効性を次のmaskで明示します。
versionのない旧archiveはv1で、maskを持ちません（v1にmaskがあれば拒否します）。
v2でmaskや理由コードが欠けた・矛盾したarchiveは保存・読込とも拒否します。

| mask | shape | 意味 | 理由コード |
|---|---|---|---|
| player_observed | P,T | 確定IDの実2D観測 | — |
| player_valid | P,T | SMPL joint0のcourt配置 | player_rejection_code |
| player_heading_valid | P,T | yaw | — |
| player_kp_3d_vis | P,T,17 | 三角測量したCOCO17 | player_kp_3d_rejection_code |
| player_smpl_valid | P,T | 配置した身体mesh | — |
| ball_3d_valid | T | 球の三角測量 | ball_rejection_code |

無効な座標は0で保存し、座標値0から有効性を推定しません。理由コード0は有効を意味します。
heading・meshは有効なrootを、3Dの人物は実2D観測を、球の3Dは方式の方針で利用可能な2 view以上の2D位置を必要とします。
有効な3Dを持つsceneは `metadata.court_reference` を必須とします。検証の正本は
`schema.validate_scene_result_arrays` です。v2では `gvhmr_aligned_*` を使いません。
rendererはmaskに従い、mesh不足frameでは有効COCO17を描画し、軌跡・速度・bounceは欠測を跨ぎません。
SLCSの教師maskは `player_valid AND player_heading_valid` と `ball_3d_valid` を必ずANDし、
2D可視性で無効な3Dのweightを復活させません。

## 同期入力と3Dの並列表示

[`MultiviewSceneRenderer`](visualization/rendering/multiview.py)は、3〜5台の同期済み入力を左、
既存rendererの3D表示を右に並べ、全パネルを同じ元frame番号で描画します。
camera IDと動画を明示的に対応付け、sceneと動画の解像度・FPS・frame数が異なる場合は拒否します。
同期推定や欠測の補間は行わず、SceneResult v2の有効性maskを既存rendererへそのまま渡します。
README用GIFのCPU生成コマンド・入力hash・目視確認結果は
[LOESS導入後の再生成・比較記録](../../knowledge/nodes/tennis_scene/000031-run-scene-ball-loess-20261026.md)を参照してください。

## フレーム別の3D人物比較ページ

`pose_comparison` は、meiji_4cam の採用中の `SceneResult` と
[`frame_pose_candidates` v1](multiview_pose/README.md) を、同じ元フレームの4視点とコート上面図に並べます。
選定は i2363 の初回 `progress.json` を使います。停止14本と、成功55本をID順に並べて seed 42 で抽出した
`clip_009`・`clip_052` が対象です。候補は割当改善後に再計測した `refine/timing/meiji4` を使います。
[再計測と成果物の版](../../knowledge/runs/run-multiview-pose-link-recall-20261124/README.md#meiji_4cam全69-clip)

リポジトリのrootで、保存済みの入力を指定して実行します。CPUで描画・圧縮します。

```bash
.venv/bin/python -m src.tennis_scene.visualization.pose_comparison \
  --dataset "$I2363_RUN/inputs/dataset" \
  --progress "$I2363_RUN/all/progress/d1df39596d4f47f1896cb535191e7040/progress.json" \
  --candidates "$POSE_OUTPUT/refine/timing/meiji4" \
  --pose-record knowledge/runs/run-multiview-pose-link-recall-20261124/meiji4.json \
  --output outputs/pose-comparison
```

`I2363_RUN` は保存済みの `i2363-all-v1`、`POSE_OUTPUT` は #2444 の `outputs/i2444` です。
`index.html` を開くと、同じディレクトリ内のMP4とポスターを閲覧できます。
`manifest.json` にclipの選定、表示した `scene_assembly`、対応する候補artifact・版、動画SHA-256、元フレームでの指標を保存します。
`--clip clip_012` などを指定すると、選定済みのclipだけを描画できます。複数指定も可能です。

旧経路は各clipの `annotations/tennis_scene/scene.json` が採用したexportとカメラ成分を照合し、
3D関節がなければ残存する2Dトラックと初回の停止理由を表示します。
採用中のシーンに有効なボール3Dがある場合は、カメラ行列を照合して同じ球を両側に表示します。
#2458・#2460で採用先が変わったら、同じコマンドの `--output` に新しいディレクトリを指定するとその時点の版で作り直せます。

動画は1888×488のH.264、CRF 21、元の3フレームごと（約19.98 fps）で、clipの時間軸全体を表示します。
容量は1動画15 MB、ページ全体200 MBが上限です。上限を超える入力では `--crf` と `--frame-step` で調整できます。
コート上面の点は有効な腰関節の中心を使い、腰が欠けた骨格は肩、最後に残存関節の中心を使います。
新経路の残差は割当済み視点に対して保存されているため、未使用視点の再投影は未計測です。
段階2の表示は `stages` に同じclip IDで動画と指標を加えられます。

## Blenderアニメーション

保存済みSceneResult v2と元動画から、入力映像がコート・選手・ボールの3Dシーンへ
切り替わるアニメーションを生成できます。書き出し・Blenderでの撮影・動画化は
[visualization/blender/export](visualization/blender/export/README.md)を参照してください。

## 他の入口

- [multiview_pose](multiview_pose/README.md): 全人物のDINO・ViTPoseと固定した動画共有sideから、フレーム別3D骨格と排他的な観測割当を作るCPU入口。
- [player_view](visualization/player_view/README.ja.md): 元動画なしで保存済みv2の人物・球・時刻・mask・頭部姿勢用資産を共通bundleへ書き出す。
- [evaluation](evaluation/README.md): 固定入力でのball 3D CPU replay・LOVO採点・比較可視化。
- [ball_events](ball_events/README.md): 保存済みSceneResult v2から打球・バウンド・ネット接触を規則で検出。
- [player pose evaluation](evaluation/player_pose/README.md): 固定SMPL再射影データによる人物検出・2D pose採点。
- [ground_truth](ground_truth/README.md): 人間提供の評価用正解の厳格な reader（再構成入力への接続なし）。
- [generate_dataset](generate_dataset/README.md): 構造化clipへの増分疑似アノテーション。
- [clip_studio](clip_studio/README.md): 同期・ラリーclip切り出し。
- [motion_alignment](motion_alignment/README.md): COCO17への身体配置。
- [synthetic_data_generation](../synthetic_data_generation/README.md): 合成データ生成。
