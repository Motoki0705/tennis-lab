---
id: run-i2633-viewpoint-narrow-s42
type: run
task: court_detection
sequence: 79
recorded_at: '2026-11-27'
title: 撮影近傍の合成512枚でコート検出を学習（seed 42）
issue: 2633
provider: codex
session: 01a21793-84f3-7432-b9eb-ca085882fa9c
date: '2026-11-26'
status: done
summary_impact: none
config:
  condition: narrow
  seed: 42
  updates: 6000
  synthetic_train: 512
  synthetic_fit_val: 64
  real_train: 0
  real_eval: real-video-v1
  batch_size: 8
  long_side: 512
  backbone: frozen DINOv3-S/16
  transformer_width: 384
  dpt_channels: 128
  precision: bf16-mixed
metrics:
  val/kp_mean_px512: 98.22494266202023
  val/kp_median_px512: 62.72598997283716
  val/kp_p90_px512: 274.7628640069441
  val/h_formed_pct: 32.92300805729633
  val/h_at_1pct_pct: 0.08952551477170993
  test/kp_mean_px512: 100.36658975919104
  test/kp_median_px512: 62.96544694277026
  test/kp_p90_px512: 280.85675365613037
  test/h_formed_pct: 33.66880146386093
  test/h_at_1pct_pct: 0.1372369624885636
  job_seconds: 5477.275222313008
repro:
  commit: 38eb91cde9d5667de2bd648b27c5cb593331c1b9
  branch: codex/parent-2627-issue-2633-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'timeout --foreground --signal=TERM --kill-after=30s 170m env PYTHONPATH=.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u knowledge/runs/run-i2633-court-viewpoint-comparison/train_and_evaluate.py
    --campaign /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2627-issue-2633-primary/court_detection/campaign/viewpoint-diversity-v1
    --checkpoint-root /home/kamimura/projects/tennis-lab/ckpt --external-asset-root
    /home/kamimura/projects/tennis-lab/third_party --condition narrow --seed 42 '
artifacts:
  run_dir: knowledge/runs/run-i2633-viewpoint-narrow-s42
  log: .training_queue/logs/0-8204305938240258536_2026-11-26T11-54-21-759741463Z_284942_i2633-viewpoint-narrow-s42.log
  output_dir: /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2627-issue-2633-primary/court_detection/campaign/viewpoint-diversity-v1/runs/narrow-s42/logs/version_0
  completion: knowledge/runs/run-i2633-viewpoint-narrow-s42/complete.json
  config: knowledge/runs/run-i2633-viewpoint-narrow-s42/config.yaml
  curves: knowledge/runs/run-i2633-viewpoint-narrow-s42/curves.png
parents: []
relations:
- to: run-court-scene-diversity-6000-v1
  rel: related
  note: 同じB00〜B03描画プールから、撮影近傍と視点区画を使って学習画像を選択した。
papers: []
tags:
- viewpoint-diversity
- fixed-budget
- real-video-v1
---

撮影近傍の条件をseed 42で実行した。B00〜B03各128枚で6,000更新を完了し、最終checkpointを読み直して同一の実画像val/testを全件採点した。学習・採点の所要時間は5477.3秒。対象コートの配分、評価順序、チェックポイントのSHA-256を固定した。

| 集合 | 画像 | KP平均 / 中央値 / P90（長辺512換算px） | H成立 | H@1% |
|---|---:|---:|---:|---:|
| val | 4,468 | 98.225 / 62.726 / 274.763 | 1,471/4,468（32.92%） | 4/4,468（0.090%） |
| test | 4,372 | 100.367 / 62.965 / 280.857 | 1,472/4,372（33.67%） | 6/4,372（0.137%） |

H@1%は成立したHの可視GT点への平均再投影誤差が画像対角長の1%以下の割合で、分母にはH不成立の画像も含む。

[実行receipt](../../runs/run-i2633-viewpoint-narrow-s42/complete.json) · [解決済み設定](../../runs/run-i2633-viewpoint-narrow-s42/config.yaml) · [両seedと視点別の比較](000084-group-i2633-court-viewpoint-diversity.md)

summary_impact: none。個別runの数値は分布比較の根拠として保持し、結果の判断は比較groupで示す。
