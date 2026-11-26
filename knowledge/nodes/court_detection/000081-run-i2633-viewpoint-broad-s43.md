---
id: run-i2633-viewpoint-broad-s43
type: run
task: court_detection
sequence: 81
recorded_at: '2026-11-27'
title: 広い視点の合成512枚でコート検出を学習（seed 43）
issue: 2633
provider: codex
session: 01a21793-84f3-7432-b9eb-ca085882fa9c
date: '2026-11-27'
status: done
summary_impact: none
config:
  condition: broad
  seed: 43
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
  val/kp_mean_px512: 105.36231389296371
  val/kp_median_px512: 52.876915459173325
  val/kp_p90_px512: 275.7842362268998
  val/h_formed_pct: 21.239928379588182
  val/h_at_1pct_pct: 0.022381378692927483
  test/kp_mean_px512: 104.53956597809677
  test/kp_median_px512: 47.699988634902525
  test/kp_p90_px512: 278.9144636902705
  test/h_formed_pct: 20.013723696248856
  test/h_at_1pct_pct: 0.022872827081427266
  job_seconds: 5064.067368593998
repro:
  commit: 38eb91cde9d5667de2bd648b27c5cb593331c1b9
  branch: codex/parent-2627-issue-2633-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'timeout --foreground --signal=TERM --kill-after=30s 170m env PYTHONPATH=.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u knowledge/runs/run-i2633-court-viewpoint-comparison/train_and_evaluate.py
    --campaign /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2627-issue-2633-primary/court_detection/campaign/viewpoint-diversity-v1
    --checkpoint-root /home/kamimura/projects/tennis-lab/ckpt --external-asset-root
    /home/kamimura/projects/tennis-lab/third_party --condition broad --seed 43 '
artifacts:
  run_dir: knowledge/runs/run-i2633-viewpoint-broad-s43
  log: .training_queue/logs/0-8204305937289569594_2026-11-26T11-54-22-710430405Z_285195_i2633-viewpoint-broad-s43.log
  output_dir: /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2627-issue-2633-primary/court_detection/campaign/viewpoint-diversity-v1/runs/broad-s43/logs/version_0
  completion: knowledge/runs/run-i2633-viewpoint-broad-s43/complete.json
  config: knowledge/runs/run-i2633-viewpoint-broad-s43/config.yaml
  curves: knowledge/runs/run-i2633-viewpoint-broad-s43/curves.png
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

広い視点の条件をseed 43で実行した。B00〜B03各128枚で6,000更新を完了し、最終checkpointを読み直して同一の実画像val/testを全件採点した。学習・採点の所要時間は5064.1秒。対象コートの配分、評価順序、チェックポイントのSHA-256を固定した。

| 集合 | 画像 | KP平均 / 中央値 / P90（長辺512換算px） | H成立 | H@1% |
|---|---:|---:|---:|---:|
| val | 4,468 | 105.362 / 52.877 / 275.784 | 949/4,468（21.24%） | 1/4,468（0.022%） |
| test | 4,372 | 104.540 / 47.700 / 278.914 | 875/4,372（20.01%） | 1/4,372（0.023%） |

H@1%は成立したHの可視GT点への平均再投影誤差が画像対角長の1%以下の割合で、分母にはH不成立の画像も含む。

[実行receipt](../../runs/run-i2633-viewpoint-broad-s43/complete.json) · [解決済み設定](../../runs/run-i2633-viewpoint-broad-s43/config.yaml) · [両seedと視点別の比較](000084-group-i2633-court-viewpoint-diversity.md)

summary_impact: none。個別runの数値は分布比較の根拠として保持し、結果の判断は比較groupで示す。
