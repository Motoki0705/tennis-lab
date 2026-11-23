---
id: run-i2444-multiview-pose-probe
type: run
task: tennis_scene
sequence: 84
recorded_at: '2026-11-24'
title: 全検出行の人物レイで向きを推定し、Meijiのフレーム別観測割当を比較
issue: 2444
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  device: cpu
  side_dataset: meiji_4cam/video_000
  side_source_stride: 12
  assignment_dataset: Meiji 3cam labelled dev clips
  assignment_side: reviewed ball, pinned by clip
  pose_joints: [5, 6, 11, 12]
  coordinate_unit: source pixel
metrics:
  meiji_4cam_clips: 69
  meiji_4cam_camera_clips_verified: 276
  meiji_4cam_detection_pose_rows_verified: 1332834
  meiji_4cam_sampled_frames: 3887
  pose_side_agreement_with_i2363: 68
  i2363_sides_available: 68
  pose_side_agreement_with_ball_shared: 69
  pose_side_min_clip_margin: 0.08228346179630264
  pose_side_video_margin: 0.1464006855724217
  meiji_3cam_frames: 3497
  meiji_3cam_detection_pose_rows_verified: 40531
  micro_pair_f1_greedy: 0.8472380952380952
  micro_pair_f1_milp: 0.8484635360311621
  micro_player_pairs_f1_greedy: 0.8676364041687566
  micro_player_pairs_f1_milp: 0.8728294746215495
repro:
  commit: f900418d36fefec21b30cc10b146a823ef5f58c8
  branch: codex/parent-1739-issue-2444-primary
  command: bash knowledge/runs/run-i2444-multiview-pose-probe/reproduce.sh
artifacts:
  run_dir: knowledge/runs/run-i2444-multiview-pose-probe
  side_ranks: knowledge/runs/run-i2444-multiview-pose-probe/side.json
  pair_f1: knowledge/runs/run-i2444-multiview-pose-probe/pair-f1.json
parents:
- run-i2363-trt-player-video
- run-court-side-video-shared-20261118
- run-i964-coco-person-features-r7-20260930
relations: []
papers: []
tags: [meiji, multiview, cpu, design_probe]
---

meiji_4camの69 clipすべてで全検出行とViTPoseの対応を読み直せた。肩・腰のレイから得た向きは、#2363で確定した68 clipとボール由来の動画共有判定に一致した。#2363の `court_side` が支持不足で停止した `clip_012` も同じ向きが首位だった。この動画では、共通のコート・drift・映像に照合したボール由来の既存判定を入力にできる。 [入力の固定と8仮説の結果](../../runs/run-i2444-multiview-pose-probe/README.md#meiji_4camの入力と向き)

Meiji 3camの既存開発4 clipでは、同じ肩・腰の候補からMILPで選ぶとmicro pair F1は0.8472から0.8485になった。1 clipではF1が下がり、目的関数の最大化と正解ラベルでの改善はclipごとに異なった。全身の検査、未使用視点への再投影、ダブルスの人物対応への採用判断は後続の評価に残る。この試作に基づく採用判断は保留する。 [割当の条件とclip別結果](../../runs/run-i2444-multiview-pose-probe/README.md#meiji-3camのフレーム別割当)・[既存pipelineの判断](../../summary.md)
