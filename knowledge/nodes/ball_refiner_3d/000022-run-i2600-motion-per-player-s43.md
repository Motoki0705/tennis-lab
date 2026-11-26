---
id: run-i2600-motion-per-player-s43
type: run
task: ball_refiner_3d
sequence: 22
recorded_at: '2026-11-26'
title: 明治4カメラ · Per Player Encoder · seed 43
issue: 2600
provider: codex
session: 01a21693-b9ec-7a42-b1bf-819187f06309
date: '2026-11-26'
status: done
summary_impact: none
config:
  epochs: 12
  gradient_clip: 1.0
  learning_rate: 0.0003
  mode: per-player
  seed: 43
  weight_decay: 0.01
  training_updates: 564
  dataset: meiji_4cam/multiview-v3
  manifest_sha256: e456f07c510b25ef4730336b18ae850090146c15cb1407c78d120ca53f4edea3
  prepared_sha256: 2d4b16b9a2f38d7097759b763ae3b7c81018233168afffef06a2c06a7bdf2bd4
  checkpoint_sha256: b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0
  validation: macro-F1, BCE, earliest epoch
  event_sigma_frames: 2.0
  tolerance_frames: 2
  peak_picking:
    min_separation: 3
    threshold: 0.5
  gpu: NVIDIA GeForce RTX 5060 Ti
  torch: 2.13.0+cu130
  cuda: '13.0'
metrics:
  test_macro_f1: 0.5629893943899297
  test_in_point_macro_f1: 0.6130115588878475
  independent_test_macro_f1: 0.5056898094872778
  independent_in_point_macro_f1: 0.5975308641975309
  test_shot_f1: 0.8245614035087719
  test_bounce_f1: 0.864406779661017
  test_net_f1: 0.0
  median_clip_ms: 7.768434996251017
  peak_cuda_mib: 117.9091796875
  added_parameters: 215552
  best_epoch: 2
  val_macro_f1: 0.6995764541587425
  val_bce: 0.06145156293430112
repro:
  commit: 976a71056e562a19f6a646c8004c4f081282246a
  branch: codex/parent-1722-issue-2600-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: test "$(git rev-parse HEAD)" = 976a71056e562a19f6a646c8004c4f081282246a
    && test -z "$(git status --porcelain -- src)" && exec env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -u -m
    src.tasks.ball_refiner_3d.scripts.post_train run --dataset /home/kamimura/projects/tennis-lab/.claude/worktrees/parent-1722-issue-2600-primary/outputs/ball_refiner_3d/post_training/prepared-e456f07c
    --manifest-sha256 e456f07c510b25ef4730336b18ae850090146c15cb1407c78d120ca53f4edea3
    --prepared-sha256 2d4b16b9a2f38d7097759b763ae3b7c81018233168afffef06a2c06a7bdf2bd4
    --checkpoint /home/kamimura/projects/tennis-lab/.worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0/checkpoints/best.ckpt
    --checkpoint-sha256 b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0
    --mode per-player --seed 43 --epochs 12 --learning-rate 0.0003 --device cuda --output
    /home/kamimura/projects/tennis-lab/.claude/worktrees/parent-1722-issue-2600-primary/outputs/ball_refiner_3d/post_training/runs/per-player-s43
artifacts:
  run_dir: knowledge/runs/run-i2600-motion-per-player-s43
  log: .training_queue/logs/0-8204322878038752469_2026-11-26T07-12-01-961247530Z_3231334_i2600-motion-per-player-s43.log
  output_dir: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/runs/per-player-s43
  predictions: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/runs/per-player-s43/predictions
  matches: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/runs/per-player-s43/matches.json
  checkpoint: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/runs/per-player-s43/best.pt
  curves: knowledge/runs/run-i2600-motion-per-player-s43/history.json
parents:
- run-i2567-net-contacts-s42
relations:
- to: run-i2600-motion-ball-only-s42
  rel: compares
papers: []
tags:
- meiji_4cam
- ball_events
- smpl_motion
---

val macro-F1 0.6996でepoch 2を選び、固定testへ一度適用した。epochごとの学習BCE・val BCE・val macro-F1は [学習ログ](../../runs/run-i2600-motion-per-player-s43/history.json) に保存した。

明治v3注釈のtest11本で全イベントmacro-F1は0.5630、ポイント内は0.6130。#2542の独立6本ではそれぞれ0.5057と0.5975だった。 [クラス別・条件別の評価](../../runs/run-i2600-motion-per-player-s43/metrics.json)

queueは `0-8204322878038752469_2026-11-26T07-12-01-961247530Z_3231334_i2600-motion-per-player-s43.job`。各予測・照合内訳と融合重みを出力ディレクトリに保存した。ハッシュと比較判断は[3方式の比較](000027-group-i2600-motion-fusion.md)に集約した。タスクsummaryの判断は同groupに基づく。
