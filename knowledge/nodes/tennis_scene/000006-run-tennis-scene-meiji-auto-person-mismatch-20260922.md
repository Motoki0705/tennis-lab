---
id: run-tennis-scene-meiji-auto-person-mismatch-20260922
type: run
task: tennis_scene
sequence: 6
recorded_at: '2026-09-23'
title: 'Meiji再実行: 同じtrack IDでも隣接コート人物を選んだため公開前に中止'
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-23'
status: failed
config:
  clip_id: video_000/clip_000
  gvhmr_track_selection: auto
  num_tracks: 2
  court_footpoint_filter_enabled: false
  queue_outcome: cancelled after identifying wrong subject
metrics:
  court_repeat_max_abs_difference: 0.0
  cam0_old_ids:
  - 1
  - 2
  cam0_new_ids:
  - 1
  - 2
  old0_new0_bbox_normalized_median: 0.0
  old1_new0_bbox_normalized_median: 12.638323499868575
  old1_new1_bbox_normalized_median: 15.975402375877962
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T191934Z/checked_entrypoint.py
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T191934Z/pipeline.json
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-auto-person-mismatch-20260922
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790104834047503676_884861_tennis-scene-cleanup-20260922T191934Z-pipeline.log
parents:
- run-tennis-scene-dino-extension-chain-20260922
relations: []
papers: []
tags:
- identity_audit
- real_clip
- failed
---

## 観測

修復済みDINO拡張でCourtと人物再構成を再実行した。Court全3030frameと校正は成立し、前回新規推論との座標差は全要素0だった。
cam0のGVHMR保存後、旧人手対応が参照するbboxと新結果を照合すると、local axis0は全frame一致したがaxis1は大きく外れた。
新旧track IDsはいずれも[1,2]であっても、元映像では新axis1が隣コート側の人物だった。
自動選択は累積bbox面積の上位であり、対象コートの選手を意味しない。

旧associationを適用せず、このタスクの実行中pipelineと待機中datasetだけを中止した。sceneは公開していない。
現状の旧GVHMR保存形式は配列のみでproducer条件を持たず、旧annotationの設定もGVHMR source=loadだったため、旧追跡条件は断定できない。

## 次の確認

現行のモデルCourtを使う既存footpoint filterを有効にして、全frameの軌跡と元映像で対象人物を確認する。
旧予測は新しい観測へコピーしない。旧bboxは人手対応の意味を照合する用途に限定する。
学習を行っていないためTensorBoardは対象外。
