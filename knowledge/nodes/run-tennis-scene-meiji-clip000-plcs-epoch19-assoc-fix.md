---
id: run-tennis-scene-meiji-clip000-plcs-epoch19-assoc-fix
type: run
title: Meiji clip_000 のcam2 player association修正後PLCS推論
provider: codex
session: 01a09a19-95aa-7ca0-828d-8f57a560125f
date: '2026-09-13'
status: done
config:
  model: ckpt/plcs/plcs-multiview-axial-split-camera-view-v2-epoch19.ckpt
  loss: inference_only
  data: data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
metrics:
  cross_view_cam0_cam2_same_player_median_m_before: 27.5408
  cross_view_cam0_cam2_same_player_median_m_after: 2.1972
  plcs_to_ankle_median_m_cam0_p0: 2.4722
  plcs_to_ankle_median_m_cam0_p1: 4.4651
  plcs_to_ankle_median_m_cam1_p0: 2.3987
  plcs_to_ankle_median_m_cam1_p1: 4.9178
  plcs_to_ankle_median_m_cam2_p0: 4.8766
  plcs_to_ankle_median_m_cam2_p1: 3.0756
repro:
  commit: 7ce569b00a74d6ce4466d5da2c1ed660909f05cf
  branch: codex/tennis-scene-camera-view-annotations
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-camera-view-annotations
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tennis_scene.scripts.generate_dataset
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt
    paths.artifact_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.cache_root=/home/kamimura/projects/tennis-lab/.cache paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    dataset_directory=tennis_multivew/processed/meiji_3cam/dataset clip_ids=[video_000/clip_000]
    overwrite=true continue_on_error=false pipeline_overrides='["court_keypoints.selector=camera_view_v2","court_reference.reference_camera=cam0","court_reference.view_half_turns=[false,false,true]","court_kp.source=load","court_kp.load_path=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/manual_court_kp_result.json","court_kp.save_result=false","gvhmr.source=load","gvhmr.load_path=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/gvhmr_result.json","gvhmr.save_result=false","player_association.source=load","player_association.load_path=tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/player_association_result.json","player_association.save_result=false","ball_detection.enabled=false","ball_detection.save_result=false","plcs.checkpoint=plcs/plcs-multiview-axial-split-camera-view-v2-epoch19.ckpt","plcs.window_size=128","plcs.window_overlap=64","plcs.sample_stride=2","plcs.save_result=false","blcs.enabled=false","blcs.save_result=false"]'
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-clip000-plcs-epoch19-assoc-fix
  log: .training_queue/logs/1789301846123677436_3529367_tennis_scene_meiji_clip000_plcs_epoch19_assoc_fix.log
  output_dir: data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene
  diagnostics: outputs/tennis_scene/meiji_3cam-video_000-clip_000-intermediates/court_position_diagnostics.json
parents:
- run-tennis-scene-meiji-clip000-plcs-e19
relations:
- to: run-tennis-scene-meiji-clip000-plcs-e19
  rel: supersedes
tags:
- tennis_scene
- plcs
- camera_view_v2
- player_association
- real_data
---

## 考察 / Findings

### 要約

3 cameraの元画像上へCourtKP、BBOX、ViTPose、GVHMR mesh、PLCS位置を重ねた結果、
従来runのcam2 player associationが逆であると判明した。cam2を入れ替えて再推論すると、
cam0–cam2間の同一player足元距離は平均中央値27.5408 mから2.1972 mへ改善した。

### アーキテクチャ詳細

モデル、手動CourtKP14、camera-view設定、保存済みGVHMRは親runと同一である。
変更点は全1010 frameのassociationだけで、canonical P0を`[cam0 local 0, cam1 local 0,
cam2 local 1]`、P1を`[local 1, local 1, local 0]`へ修正した。PLCSはstride 2、
128-frame window、64-frame overlapで再実行した。

### メトリクスの解釈

CourtKP homographyでCOCO-17両足首中点をcourt XYへ逆投影し、PLCS位置との差を実クリップ上の
診断値とした。修正後の中央値はP0がcam0/1/2で2.4722/2.3987/4.8766 m、P1が
4.4651/4.9178/3.0756 mである。associationの破綻は解消したが、絶対位置精度は未解決である。

### アーキテクチャ⇄メトリクスの因果考察

cam2はphysical courtの反対側にあり、同じ画像内local indexは同じcanonical playerを意味しない。
従来のidentity associationは半回転したcamera viewをまたいで人物を混同し、PLCS入力のview間
対応を壊していた。修正後も予測が両baselineからnet側へ約2–5 m縮む傾向が残る。これは観測上の
系統誤差であり、checkpointの実映像domain gapまたはCourtKP/pose特徴に対する奥行き回帰の
平均化が原因というのは仮説である。

### 既存実験との比較

親run `run-tennis-scene-meiji-clip000-plcs-e19` はframe 1/300/700/900の見た目だけからidentity
associationを採用したが、court homographyで足元を共通座標へ戻す検査をしていなかった。
本runはその結論をsupersedeし、camera間の幾何整合性を定量条件として使う必要を示した。

### 次に有効な実験

実データで足首逆投影位置を補助教師または評価指標にし、PLCSのbaseline方向の縮みを測る。
またmanual association UIへ、各候補対応のcross-view court距離を表示して25 m級の誤対応を
保存前に拒否する検証を追加する。
