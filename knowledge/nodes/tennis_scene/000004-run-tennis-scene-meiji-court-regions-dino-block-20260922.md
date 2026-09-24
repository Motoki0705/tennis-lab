---
id: run-tennis-scene-meiji-court-regions-dino-block-20260922
type: run
task: tennis_scene
sequence: 4
recorded_at: '2026-09-23'
title: Meiji全3030frameのCourt成功と旧DINO拡張による停止
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-23'
status: failed
config:
  clip_id: video_000/clip_000
  camera_ids:
  - cam0
  - cam1
  - cam2
  frames: 1010
  court_checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  region_search: image_only_fixed_grid
  all_automatic_stages: execute
metrics:
  cam0:
    all_visible_frames: 1010
    visible_points: 14140
    mean_error_px: 9.927464882672615
    median_error_px: 8.341543394414227
    p95_error_px: 18.52688945349262
    first_frame_mean_px: 10.782743083086343
  cam1:
    all_visible_frames: 1010
    visible_points: 14140
    mean_error_px: 13.129072637822086
    median_error_px: 10.388039341356777
    p95_error_px: 28.20111653932178
    first_frame_mean_px: 16.645575434270395
  cam2:
    all_visible_frames: 1010
    visible_points: 14140
    mean_error_px: 8.374166851061755
    median_error_px: 8.719633601111209
    p95_error_px: 12.389126399105175
    first_frame_mean_px: 8.0536277785548
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/build/lib.linux-x86_64-cpython-311
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/checked_entrypoint.py
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/pipeline.json
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-court-regions-dino-block-20260922
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790096996294984088_57107_tennis-scene-cleanup-20260922T170847Z-pipeline.log
parents:
- run-court-meiji-model-only-regions-20260922
relations: []
papers: []
tags:
- real_clip
- court_calibration
- dino_extension
- failed
---

## 観測

Courtを3camera×1010frameで新規推論し、全3030frameでKP14が画像内有効となった。
manual点との全有効点平均距離はcam0 9.93px、cam1 13.13px、cam2 8.37px。
frame 0を使うcamera_view_v2の3camera校正も通過した。Courtは約116分を要した。

直後のDINO tracking最初のCUDA forwardで`Unrecognized tensor type ID: PythonDispatcher`が発生し、scene生成前に停止した。
後続datasetジョブは先行scene未生成のguardで止まり、dataset本体を実行済みとは扱わない。
これまで確認していた拡張のimportとsm_120 cubinの存在だけでは、現行Torchでの演算互換性を確認したことにならなかった。

## 解釈と次の確認

Courtについては少数frame probeに加えて全区間の幾何推定成立を確認できた。manual参照は静的な2D点であり、3D精度の証明ではない。
DINOの失敗は別の実行境界で、後続の拡張再buildとforward/backward確認で切り分ける。
学習は行っていないためTensorBoard曲線は対象外。最終評価の両入口は、修復後の拡張を明示して全自動段を先頭から再生成する。
