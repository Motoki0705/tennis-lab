---
id: run-i2600-motion-set-attention-s43-r2
type: run
task: ball_refiner_3d
sequence: 34
recorded_at: '2026-11-27'
title: 明治4カメラ · Set Attention Block · seed 43（coverage修正後）
issue: 2600
provider: codex
session: 01a21693-b9ec-7a42-b1bf-819187f06309
date: '2026-11-27'
status: done
summary_impact: none
config:
  epochs: 12
  gradient_clip: 1.0
  learning_rate: 0.0003
  mode: set-attention
  seed: 43
  weight_decay: 0.01
  training_updates: 564
  dataset: meiji_4cam/multiview-v3
  manifest_sha256: 790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a
  prepared_sha256: e0c54222f9cfbfde278956e4438d158fe9fa4a17e92ec791b607bac396c24677
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
  test_macro_f1: 0.6732909491530181
  test_in_point_macro_f1: 0.7506060606060606
  independent_test_macro_f1: 0.4718614718614719
  independent_in_point_macro_f1: 0.5559038662486938
  test_bounce_f1: 0.896551724137931
  test_net_f1: 0.2857142857142857
  test_shot_f1: 0.8376068376068376
  median_clip_ms: 7.8539359965361655
  peak_cuda_mib: 150.8515625
  added_parameters: 779136
  best_epoch: 1
  val_macro_f1: 0.6747522170057382
  val_bce: 0.06143482029438019
repro:
  commit: 3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f
  branch: codex/parent-1722-issue-2600-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: test "$(git rev-parse HEAD)" = 3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f
    && test -z "$(git status --porcelain -- src)" && exec env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -u -m
    src.tasks.ball_refiner_3d.scripts.post_train run --dataset /home/kamimura/projects/tennis-lab/.claude/worktrees/parent-1722-issue-2600-primary/outputs/ball_refiner_3d/post_training/coverage-r2/prepared-790f0501
    --manifest-sha256 790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a
    --prepared-sha256 e0c54222f9cfbfde278956e4438d158fe9fa4a17e92ec791b607bac396c24677
    --checkpoint /home/kamimura/projects/tennis-lab/.worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0/checkpoints/best.ckpt
    --checkpoint-sha256 b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0
    --mode set-attention --seed 43 --epochs 12 --learning-rate 0.0003 --device cuda
    --output /home/kamimura/projects/tennis-lab/.claude/worktrees/parent-1722-issue-2600-primary/outputs/ball_refiner_3d/post_training/coverage-r2/runs/set-attention-s43
artifacts:
  run_dir: knowledge/runs/run-i2600-motion-set-attention-s43-r2
  log: .training_queue/logs/0-8204305170772348783_2026-11-26T12-07-09-227651216Z_435125_i2600-motion-set-attention-s43-r2.log
  output_dir: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2/runs/set-attention-s43
  predictions: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2/runs/set-attention-s43/predictions
  matches: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2/runs/set-attention-s43/matches.json
  checkpoint: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2/runs/set-attention-s43/best.pt
  curves: knowledge/runs/run-i2600-motion-set-attention-s43-r2/history.json
parents:
- run-i2567-net-contacts-s42
relations:
- to: run-i2600-motion-set-attention-s43
  rel: supersedes
  note: coverage外の既知イベントを教師と採点に含めた再実験。
- to: run-i2600-motion-ball-only-s42-r2
  rel: compares
papers: []
tags:
- meiji_4cam
- ball_events
- smpl_motion
---

val macro-F1 0.6748でepoch 1を選び、固定testへ適用した。 [全12 epochの学習履歴](../../runs/run-i2600-motion-set-attention-s43-r2/history.json)

明治v3注釈のtest11本で全イベントmacro-F1は0.6733、ポイント内は0.7506。#2542の独立ラベル6本ではそれぞれ0.4719と0.5559だった。 [クラス別・条件別の評価](../../runs/run-i2600-motion-set-attention-s43-r2/metrics.json)

固定入力とqueueの完了記録を照合し、予測と照合結果を出力ディレクトリに保存した。タスクsummaryの判断は3方式の比較に基づく。 [queue完了記録](../../runs/run-i2600-motion-set-attention-s43-r2/queue-completion.json) · [3方式の比較](000027-group-i2600-motion-fusion.md)
