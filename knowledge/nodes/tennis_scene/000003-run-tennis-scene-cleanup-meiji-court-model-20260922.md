---
id: run-tennis-scene-cleanup-meiji-court-model-20260922
type: run
task: tennis_scene
sequence: 3
recorded_at: '2026-09-23'
title: 'Meiji clip_000: 既定Court推論の再生成と校正条件の不成立'
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-22'
status: failed
config:
  clip_id: video_000/clip_000
  camera_ids:
  - cam0
  - cam1
  - cam2
  frames: 1010
  court_checkpoint_sha256: dd3a396841097e60ff1bc0eabcf7b911e97685e251bf8cc441c100b17276e816
  court_inference: per_frame_hybrid
  calibration_frame: 0
  reference_camera: cam0
  view_half_turns:
  - false
  - false
  - true
metrics:
  complete_frames_cam0: 1
  complete_frames_cam1: 0
  complete_frames_cam2: 21
  common_complete_calibration_frames: 0
  raw_frame0_mean_error_px:
    cam0: 701.3568459131246
    cam1: 180.88906481429157
    cam2: 47.78015657335855
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/build/lib.linux-x86_64-cpython-311
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T131222Z/checked_entrypoint.py
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T131222Z/pipeline.json
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-cleanup-meiji-court-model-20260922
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790082742756457010_320195_tennis-scene-cleanup-20260922T131222Z-pipeline.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/responsibility_cleanup/20260922T131222Z-pipeline
parents: []
relations: []
papers: []
tags:
- real_clip
- court_calibration
- pipeline_integration
- failed
---

## 観測結果

責務分離後の通常パイプラインを、Meiji `video_000/clip_000` の3カメラ・1010 frame全区間で実行した。
Courtの3030 frame推論と結果保存は完了したが、frame 0の14点が揃わず、reference校正で明示エラーとなった。
全frameでもcam0の完全14点は1件、cam1は0件、cam2は21件で、3camera共通の校正frameは存在しない。
したがって校正frame番号の変更だけでは続行できない。GVHMR・PLCS・ball・BLCSとscene保存には到達していない。
後続datasetジョブも先行scene未生成の事前条件で停止し、dataset本体や可視化を実行済みとは扱わない。

frame 0のraw KPを既存manual点の同じchannelと比較した平均画素距離はcam0 701.36、cam1 180.89、cam2 47.78。
cam0はフェンス・黒帯側へ大きく外れている。点順を最適置換する診断でもcam0は587.87pxで、
単一のpoint permutationだけで説明できない。最適置換値は診断であり、通常精度として採用しない。
手動Court点を入力にするCPU校正preflightは成立したが、ユーザー指定によりこのrunには使用していない。

## 再現条件と限界

[証拠](../../runs/run-tennis-scene-cleanup-meiji-court-model-20260922/evidence/)に設定、入力SHA、raw比較と画像を保存した。
実行時に記録されたCourt checkpoint SHAは実行前のdual SHAと一致した。
Court・外注ball注釈は評価用で、推論の入力にはしていない。manual Courtとの2D比較であり、3D実測GTはない。
TensorBoard曲線は学習を行っていないため対象外。

[KP＋LINE下流移行](../court_detection/000031-run-court-hybrid-downstream-migration.md)は入力契約の統一であり、
実写頑健性の確立ではないという従来の限界と整合する。本結果だけで別会場の精度は判断しない。

## 次の作業

ユーザーは2026-09-23にCourtモデル・後処理の改善を追加し、モデル出力で続行する方針を指定した。
manual入力へのfallbackは行わず、既存のMeiji probe結果を参照してcheckpoint・前処理・後処理を切り分ける。
通常scene生成、dataset生成、3D/タスク別可視化の完走と評価は、その後のrunで確認する。
