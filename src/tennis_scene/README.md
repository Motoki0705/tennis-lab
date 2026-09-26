# Tennis Scene

同期済みマルチカメラ動画から `SceneResult` を組み立て、NPZとmetadataへ保存します。
入力動画の準備、sceneの増分生成、保存結果の可視化を担当します。固定カメラを前提とします。
SLCS向けの特徴抽出・学習split準備は [SLCS生成ガイド](../tasks/slcs/generate_dataset/README.md)、
再構成sceneからの合成学習データ生成は [synthetic_data_generation](../synthetic_data_generation/README.md) が担当します。

動画注釈のリクエスト準備と添付方法は [chat_annotation](chat_annotation/README.md) を参照してください。

## パスと公開入口

`scripts/` の実行入口は以下の5つです。コマンドはrepoルートの `.venv/bin/python -m src.tennis_scene.scripts.<入口>` で実行します。
このREADMEを5入口の入出力構造の正本とします。

| 表記 | 設定 | 既定root | 用途 |
|---|---|---|---|
| DATA | `paths.data_root` | `data/` | 元動画、構造化dataset、dataset付属scene |
| ARTIFACT | `paths.artifact_root` | `outputs/` | 保存済みscene・中間結果の読み込みとstage別結果の保存 |
| OUTPUT | `paths.output_root` | `outputs/` | 単発scene、可視化、実行ログ |
| CHECKPOINT | `paths.checkpoint_root` | `ckpt/` | Court・DINO detector・PLCS・BLCS・ballの重み |
| EXTERNAL_ASSET | `paths.external_asset_root` | `third_party/` | GVHMR関連重み、DINO source、描画用regressor |

引数のパスは各rootからの相対fragmentです。`data/` や `outputs/` を重ねず、絶対パスは
`paths.*_root` に指定します。専用worktreeでは入力rootをメインrepoの絶対パスへ明示します。
共通のrun命名規則は [タスク出力規約](../tasks/OUTPUTS.md) を参照してください。

## 1. clip_studio — 入力動画を同期・切り出す

入力は必ず次の構造です。`video_id` は `video_000` のような3桁以上の数字、camera番号は
`cam0` から欠番なしとします。直下の別名MP4、番号の欠番、標準外の階層はエラーです。
`tennis_multivew` は現行のディレクトリ名をそのまま使用します。

```text
<DATA>/tennis_multivew/
├── raw/<dataset_id>/<video_id>/
│   ├── cam0.mp4
│   ├── cam1.mp4
│   └── ...
└── processed/<dataset_id>/                     # 以下を作成・更新
    ├── projects.json                          # 同期offset・clip編集状態
    └── dataset/
        ├── dataset.json                       # clip一覧
        └── videos/<video_id>/clips/<clip_name>/
            ├── clip.json                      # camera順・動画・時間の契約
            └── media/<camera_id>.mp4           # 同期済み動画
```

```bash
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  source_directory=tennis_multivew/raw/meiji_3cam/video_000
```

sourceからproject・datasetの保存先を一意に導出します。再起動は保存済みprojectを復元します。
GUIの書き出しは、編集内容と全動画を検証できた完成clipを再利用し、不完全・不一致の出力を拒否します。
再出力は `export.overwrite=true` で明示します。操作・同期・キャンセルの詳細は
[Clip Studio](clip_studio/README.md) を参照してください。ヘッドレス書き出し・旧layout移行CLIは提供しません。

## 2. generate_dataset — datasetの各clipへsceneを追加する

入力は上記の `dataset.json`、各 `clip.json`、`media/` を持つ構造化datasetです。
`clip_ids` は `<video_id>/<clip_name>` 形式で、省略すると全clipを対象にします。

```text
<DATA>/<dataset_directory>/
├── dataset.json
└── videos/<video_id>/clips/<clip_name>/
    ├── clip.json
    ├── media/<camera_id>.mp4
    └── annotations/
        ├── tennis_scene/
        │   ├── scene.npz
        │   ├── scene.metadata.json             # 必須sidecar
        │   ├── annotation.json                 # 完成マーカー
        │   └── pipeline_config.yaml            # 解決済み生成設定
        └── tennis_scene.failure.json           # 失敗した場合
```

```bash
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  dataset_directory=tennis_multivew/processed/meiji_3cam/dataset \
  clip_ids='[video_000/clip_000]' \
  'pipeline_overrides=["court_reference.reference_camera=cam0","court_reference.view_half_turns=[false,false,true]","court_kp.save_result=false","gvhmr.save_result=false","player_association.save_result=false","ball_detection.save_result=false","plcs.save_result=false","blcs.save_result=false"]'
```

`pipeline.yaml` を共通設定として使い、差分は `pipeline_overrides` で指定します。
CLIでpipeline_overridesを指定するとリスト全体を置換します。例では中間保存を無効にする指定も含めています。
実clipの動画とcamera順は `clip.json` から取得します。保存する `pipeline_config.yaml` は共通設定であり、
実際に使用した動画の絶対パスとcamera順は `scene.metadata.json` に記録します。
モデル生成には既定checkpointと3〜4台の同期動画が必要です。選手対応の既定は手動UIです。
保存済み対応を使う場合は `player_association.source=load` とARTIFACT相対の `load_path` を明示し、
再推論後のlocal player軸が同じ人物を表すことを確認してください。
`gvhmr.track_selection=auto` は累積bbox面積の上位を選ぶため、隣接コートの人物が入る場合があります。
DINOでは `gvhmr.court_footpoint_filter.enabled=true` で推定コート周辺へ検出を限定できます。
track IDが同じでも人物一致を意味しないため、保存済み対応の適用前に軌跡と元映像を照合します。

完成マーカーがあるclipは既定でskipし、再生成は `overwrite=true` に限定します。
不完全な生成ディレクトリは自動採用しません。失敗理由を保存し、CLIは非0で終了します。
`continue_on_error=false` は最初の失敗で停止します。既定ではstage別の自動結果保存を無効化し、
scene一式だけをclipへ追加します。詳しい公開トランザクションは
[生成処理](generate_dataset/README.md) を参照してください。

## 3. run_pipeline — 指定動画から単発sceneを保存する

動画の格納階層は任意ですが、すべてDATA配下の既存ファイルで、FPS・フレーム数・解像度が一致し、
同期済みである必要があります。`video_paths` と `camera_ids` は同じ順・同じ数にします。

```text
<DATA>/<任意のclip階層>/cam0.mp4, cam1.mp4, cam2.mp4

<OUTPUT>/<output_directory>/
├── <output_name>.npz
├── <output_name>.metadata.json
└── hydra/                                     # 実行ログ

<ARTIFACT>/<output_directory>/                  # 各stageのsave_result=trueの場合
├── court_kp_result.json
├── gvhmr_result_cam0.json, gvhmr_result_cam1.json, ...
├── player_association_result.json              # 対応を新規作成・保存した場合
├── ball_detection_result.json
├── plcs_result.json
└── blcs_result.json
```

```bash
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[samples/cam0.mp4,samples/cam1.mp4,samples/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' \
  output_name=scene \
  'court_reference.view_half_turns=[false,false,true]'
```

既定の `output_directory` は `tennis_scene/generate/pipeline/<run-id>`、`output_name` は
`tennis_clip` です。各stageの `output_path` は個別指定もできます。
単発保存にはdatasetの完成マーカーはありません。既存ファイルを指定すると保存時に上書きするため、
比較実行には別runを使います。`source=execute` は推論、`source=load` は保存済みstage結果を読み込みます。

全画面ではコートが小さい・複数面が写る固定カメラは、`court_kp.region_search.enabled=true` で
画像からの領域探索を明示できます。`court_kp.frame_index` のモデル出力からcameraごとに領域を決め、
その領域で全frameを再推論します。選択領域・候補の採否・各frameのcrop/native/元画像サイズは
`court_kp_result.json` のdiagnosticsへ記録します。checkpointは `court_kp.checkpoint` で指定します。
領域選択と座標変換の契約は [Court推論](../tasks/court_detection/README.md#共通推論と幾何補正) を参照してください。
`generate_dataset` では同じ指定を `pipeline_overrides` に入れます。

## 4. visualization — 保存sceneの3D動画を作る

必須入力はARTIFACT相対の `input` と同名の `.metadata.json` です。SMPL表示・skeleton表示とも、
現行入口は `smpl_vertices_local`、`smpl_global_orient`、選手位置・yawを必要とします。
描画用facesはDATA相対の `assets.smpl_faces`、regressorはEXTERNAL_ASSET相対です。

```text
<ARTIFACT>/<input>.npz
<ARTIFACT>/<input>.metadata.json
<OUTPUT>/<output>                              # MP4/GIFなど
<OUTPUT>/<preview_output>                      # output=nullかつdisplay=falseの場合
```

```bash
.venv/bin/python -m src.tennis_scene.scripts.visualization \
  input=tennis_scene/generate/pipeline/<run-id>/scene.npz \
  output=tennis_scene/visualize/visualization/<run-id>/scene.mp4
```

dataset内のsceneは `paths.artifact_root=/absolute/path/to/data` とし、
`input=tennis_multivew/processed/<dataset>/dataset/videos/<video>/clips/<clip>/annotations/tennis_scene/scene.npz`
で指定します。出力rootは別途OUTPUTです。`display=true` は画面表示を有効にし、
`output=null display=false` は開始frameのPNGを保存します。保存先の既存ファイルは上書きされます。

## 5. visualize_tasks — stageごとの結果を描画する

ARTIFACT相対の `scene_path` と必須metadata、DATA相対の `video_paths` を明示します。
scene metadataから元動画を自動選択しません。2D描画は `video_paths[0]` とsceneの先頭cameraを使用します。

```text
<OUTPUT>/<output_directory>/
├── ball_detection_viz.mp4
├── court_kp_viz.mp4
├── gvhmr_viz.mp4
├── plcs_viz.mp4
├── gvhmr_alignment_viz.mp4
└── blcs_viz.mp4
```

```bash
.venv/bin/python -m src.tennis_scene.scripts.visualize_tasks \
  scene_path=tennis_scene/generate/pipeline/<run-id>/scene.npz \
  'video_paths=[samples/cam0.mp4]' \
  output_directory=tennis_scene/visualize/visualize_tasks/<run-id>
```

`tasks` で出力対象を選べます。指定stageの配列がない場合はエラーにします。
3Dだけを描画する場合も `video_paths` 設定は必要ですが、元動画のdecodeは2Dタスク選択時だけです。
同名の出力動画は上書きされます。

## ログ、スキーマ、モデル契約

HydraログはOUTPUT内の `tennis_scene/<generate|visualize>/<入口の設定名>/<run-id>/hydra/` に置きます。
`run_pipeline` の設定名は `pipeline` です。`run_pipeline` と `visualize_tasks` では
明示した `output_directory` がログ保存先の基準にもなります。

`schema.SceneResult` と `archive.save_scene_result/load_scene_result` がsceneの唯一の定義・I/Oです。
metadata sidecarの欠落はエラーです。ボールなど無効にしたstageの配列は省略されるため、
下流は必要な配列を検証します。SLCSに必要な追加契約はSLCS側が所有します。
`camera_view_v2` の保存済みsceneは [camera-local観測契約](../tasks/base/generate_dataset/README.md)
も満たす必要があります。旧reference順sceneは再生成が必要です。

### SceneResult v2

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
heading・meshは有効なrootを、3Dの人物は実2D観測を、球の3Dは2 view以上の観測を必要とします。
有効な3Dを持つsceneは `metadata.court_reference` を必須とします。検証の正本は
`schema.validate_scene_result_arrays` です。v2では `gvhmr_aligned_*` を使いません。
rendererはmaskに従い、mesh不足frameでは有効COCO17を描画し、軌跡・速度・bounceは欠測を跨ぎません。
SLCSの教師maskは `player_valid AND player_heading_valid` と `ball_3d_valid` を必ずANDし、
2D可視性で無効な3Dのweightを復活させません。

ViTPoseの生ヒートマップピークは確率ではなく1を超えることがあります。sceneと下流推論に渡す
姿勢visibilityは有限性を検証して `[0,1]` へ飽和させ、元の範囲・飽和件数を
`metadata.pose_visibility_conversion` に記録します。GVHMRのstage保存結果は生値を保持します。

BallのRGB入力はpredictorへ未正規化の `[0,1]` で渡し、重みに保存されたImageNet正規化をpredictor内で適用します。
`ball_detection.normalize_imagenet` は保存設定と一致することを確認する指定で、mean/stdはcheckpointから読みます。
設定の不一致や保存情報の不足はエラーです。

`pipeline/` は各task-owned predictorを統合します。既定のCourt・PLCS・BLCSは `camera_view_v2` で、
PLCS/BLCSのwindowは128 frameです。camera ID、reference camera、各viewの半回転を明示します。
旧physical順のartifactを使う場合は契約と対応checkpointを明示し、camera-view順と混在させません。

PLCS配置とGVHMRの整列済み配置は別フィールドに保存します。
座標系・整列契約は [motion_alignment](motion_alignment/README.md) を参照してください。
コート座標はXY平面・Z-up、SMPL人体座標はY-upです。
