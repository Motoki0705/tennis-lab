---
id: run-multiview-pose-meiji-20261124
type: run
task: tennis_scene
sequence: 99
recorded_at: '2026-11-24'
title: 全身のフレーム別3D骨格と排他割当をMeijiで評価
issue: 2444
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  device: cpu
  frame_axis: native source frames
  association: geometric pose hypotheses and per-conflict MILP
  side_meiji4: pinned ball-based shared video decision
  side_meiji3: fixed reviewed-ball clip reference for evaluation
  meiji4_workers: 4
metrics:
  meiji3_source_frames: 3497
  meiji3_micro_pair_f1: 0.8049526831756106
  meiji3_probe_micro_pair_f1: 0.8484635360311621
  meiji3_player_only_micro_pair_f1: 0.8257760725497035
  meiji3_holdout_scoreable_person_frames: 15458
  meiji3_holdout_eligible_person_frames: 19860
  meiji3_holdout_scoreable_joints: 251343
  meiji3_holdout_eligible_joints: 332112
  meiji3_holdout_p50_px: 9.591305515810095
  meiji3_holdout_p90_px: 27.814713179557703
  meiji4_clips: 69
  meiji4_source_frames: 46228
  meiji4_populated_frames: 46228
  meiji4_candidates: 348542
  meiji4_assigned_detection_rows: 1146520
  meiji4_detection_rows: 1332834
  meiji4_wall_seconds: 329.0027961980086
repro:
  commit: 85bc35af29d6f6623359123f5f64ea20d9259617
  branch: codex/parent-1739-issue-2444-primary
  command: bash knowledge/runs/run-multiview-pose-meiji-20261124/reproduce.sh
artifacts:
  run_dir: knowledge/runs/run-multiview-pose-meiji-20261124
  meiji3: knowledge/runs/run-multiview-pose-meiji-20261124/meiji3.json
  meiji4: knowledge/runs/run-multiview-pose-meiji-20261124/meiji4.json
parents:
- run-i2444-multiview-pose-probe
- run-i2363-trt-player-video
- run-court-side-video-shared-20261118
- run-i964-coco-person-features-r7-20260930
relations: []
papers: []
tags: [meiji, multiview, cpu, frame_pose]
---

全身の関節と骨長を検査して排他割当したところ、Meiji 3camの4開発clipでmicro pair F1は肩・腰試作の0.8485から0.8050になった。FPは1,102→906、FNは4,500→5,978。3視点のLOVOでは251,343 / 332,112関節を採点でき、再投影誤差は中央値9.59 px、P90 27.81 pxだった。 [採点と未使用視点の条件](../../runs/run-multiview-pose-meiji-20261124/README.md)

meiji_4camではボール由来の共有判定をpinして69 clip・46,228元frameすべてで候補を保存した。4 CPUプロセスでの入力照合から公開確認まで329.00秒。今回の新経路はフレーム別の候補までを対象とし、標準pipelineの採用判断は据え置く。 [全69 clipの結果](../../runs/run-multiview-pose-meiji-20261124/meiji4.json)
