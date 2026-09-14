# `src/tennis_scene/generate_dataset`

`clip_studio` が追記した構造化データセットを読み、`TennisSceneOrchestrator` の出力をクリップ単位の疑似アノテーションとして保存します。データセット全体を一度に作り直さず、`annotation.json` がないクリップだけを処理します。

## 生成物

```text
<dataset_root>/
├── dataset.json
└── videos/<video_id>/clips/<clip_name>/
    ├── clip.json
    ├── media/<camera_id>.mp4
    └── annotations/tennis_scene/
        ├── scene.npz
        ├── scene.metadata.json
        ├── annotation.json
        └── pipeline_config.yaml
```

`scene.npz` は `SceneResult` のcanonical schemaです。`annotation.json` は必須配列のshape/dtypeと入力 `clip.json` のSHA-256を持つ完成マーカーで、これがないディレクトリは完成済みとして扱いません。既存の完成結果は既定でskipし、再生成は `overwrite=true` でのみ行います。失敗は `annotations/tennis_scene.failure.json` に記録し、CLIは非0で終了します。

## 実行

```bash
# まだ生成されていない全クリップ
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  dataset_directory=tennis_multivew/processed/meiji_3cam/dataset

# 一部だけ明示選択
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  dataset_directory=tennis_multivew/processed/meiji_3cam/dataset \
  clip_ids='[video_000/clip_000]'
```

モデル・checkpoint設定は既存の `configs/pipeline.yaml` を直接読み、`configs/generate_dataset.yaml` はdataset生成時の差分だけを `pipeline_overrides` として保持します。これによりパイプライン設定を二重管理しません。

### camera-view checkpointで実クリップを生成

`camera_view_v2` checkpointは、各cameraがphysical courtに対して半回転しているかを
`court_reference.view_half_turns`で明示し、非回転側のstable camera IDを
`court_reference.reference_camera`に指定します。以下はMeiji 3-camera clipを、手動CourtKP14と
PLCSだけで生成する例です。初回はplayer association UIが開き、選択結果をclip配下へ保存します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  paths.artifact_root=data \
  dataset_directory=tennis_multivew/processed/meiji_3cam/dataset \
  clip_ids='[video_000/clip_000]' \
  continue_on_error=false \
  'pipeline_overrides=["court_keypoints.selector=camera_view_v2","court_reference.reference_camera=cam0","court_reference.view_half_turns=[false,false,true]","court_kp.source=load","court_kp.load_path=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/manual_court_kp_result.json","court_kp.save_result=false","gvhmr.court_footpoint_filter.enabled=true","gvhmr.court_footpoint_filter.sideline_margin_m=1.0","gvhmr.court_footpoint_filter.baseline_margin_m=5.0","gvhmr.save_result=true","gvhmr.output_path=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/gvhmr_result.json","player_association.source=execute","player_association.save_result=true","player_association.output_path=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/player_association_result.json","ball_detection.enabled=false","ball_detection.save_result=false","plcs.checkpoint=plcs/plcs-multiview-axial-split-camera-view-v2-epoch19.ckpt","plcs.window_size=128","plcs.window_overlap=64","plcs.sample_stride=2","plcs.save_result=false","blcs.enabled=false","blcs.save_result=false"]'
```

`sample_stride=2`は59.94 fpsのsource timelineを約29.97 fpsで推論し、positionは線形、yawは
単位heading vectorで補間して全source frameへ戻します。CourtKP契約、reference camera、camera側、
typed reference metadataのいずれかが欠ける場合は推論前に失敗します。
`court_footpoint_filter`はCourtKP14から対象コートの画像polygonを作り、指定した実寸marginの
外側に足元がある隣接コートの人物をDINO追跡前に除外します。DINO以外で有効化した場合や、
基準frameのCourtKP14が欠ける場合は明示的に失敗します。
`gvhmr.output_path`はcameraごとに`_cam0`、`_cam1`、`_cam2`を付けて保存されます。
この3ファイルと`player_association_result.json`を再利用する場合は、完成済みannotationの
skipを避けるため`overwrite=true`を指定し、`gvhmr.source=load`、`gvhmr.load_path=<同じbase path>`、
`player_association.source=load`、`player_association.load_path=<保存したpath>`へ切り替えます。

player association UIでは、`n`/`p`で表示frameを移動し、`s <frame>`で時間区間を分割します。
割当は`a <segment> <canonical-player> <camera-id-or-index> <local-player>`で変更し、
`save`で現在の全区間を検証・保存して推論を続行します。

生成済みsceneのMP4可視化は次です。

```bash
.venv/bin/python -m src.tennis_scene.scripts.visualization \
  paths.artifact_root=data \
  input=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene/scene.npz \
  output=tennis_scene/meiji_3cam-video_000-clip_000-plcs-epoch19.mp4
```

BLCSのreprojection lossに必要な実カメラparameterはこのschemaでは捏造しません。実データを既存のsimulation dataset loaderへ混ぜる処理は、キャリブレーション契約とsplit方針を決めた後の別スコープです。
