---
task: slcs
sequence: 125
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-pilot-no-ball-smooth-ball8-e60-v1
type: run
title: 'SLCS pilot 60epoch: no-smoothからball位置weightだけ8倍、改善せず'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: SLCSLoss, ball_position_smoothness_weight=0, ball_position_weight=8
  data: slcs/real_rgb_pilot_v2
  max_epochs: 60
  seed: 42
metrics:
  player_position_error_m: 3.041474
  player_position_error_median_m: 2.004761
  player_angular_error_deg: 47.33382
  player_angular_error_median_deg: 45.036285
  player_position_accuracy_0.3m: 0.0
  player_position_accuracy_0.5m: 0.006849
  player_position_accuracy_1.0m: 0.091324
  player_position_accuracy_2.0m: 0.497717
  player_angle_accuracy_10deg: 0.050228
  player_angle_accuracy_15deg: 0.079909
  player_angle_accuracy_30deg: 0.26484
  player_position_pred_b_m: 1.09359
  player_rotation_pred_b_deg: 23.834661
  player_position_conf_error_corr: 0.472021
  player_rotation_conf_error_corr: -0.361384
  ball_position_error_m: 4.916261
  ball_position_error_median_m: 4.261795
  ball_position_accuracy_0.3m: 0.0
  ball_position_accuracy_0.5m: 0.0
  ball_position_accuracy_1.0m: 0.077844
  ball_position_accuracy_2.0m: 0.197605
  ball_position_pred_b_m: 2.438958
  ball_position_conf_error_corr: 0.525207
  scene_position_error_m: 3.978868
repro:
  commit: 7621307bf292f5b06bd6d3fe2e0cf90a7ee27985
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot_no_ball_smooth
    loss.ball_position_weight=8.0 run.output_dir=slcs/train/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-ball8-e60-v1
  predictions: knowledge/runs/run-slcs-pilot-no-ball-smooth-ball8-e60-v1/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001/logs/version_0
  curves: knowledge/runs/run-slcs-pilot-no-ball-smooth-ball8-e60-v1/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001/logs/version_0
parents: [run-slcs-pilot-no-ball-smooth-e60-takeover-v1, run-slcs-ball-gradient-probe-v1]
relations:
- {to: run-slcs-pilot-no-ball-smooth-e60-takeover-v1, rel: compares}
tags: [slcs, pilot, loss-weight, single-intervention]
---

## 考察 / Findings

### 要約

no-smooth pilotからball位置Smooth L1のweightだけ1→8として60epoch完走した。
train ball終端6.4879mはweight1の6.4757mを改善せず、validation選定評価も改善しなかったため不採用。

### アーキテクチャ詳細

旧7clip、88train窓、batch16、360更新、seed42、augmentation、warmup20を固定した単一係数比較。
ball NLL weight0.5、player各損失は変更していない。既存勾配診断を根拠とした探索である。
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)は動的な勾配調整を扱うが、本runはその再現ではなく
勾配配分という問題設定を参考にした固定係数の仮説検証であり、8という値自体は論文由来ではない。

### メトリクスの解釈

frontmatter/pred_test.npzはrunnerの終端last重みの自動test出力で、選定根拠には使わない。
train player13.8098→2.0798m、ball8.7676→6.4879m。val終端player2.9131m、ball7.0130m。
validation scene monitor最小4.955067mのepoch56を選んだCPU評価を別nodeに保存した。
curves.pngでもball頭打ちが残り、係数を上げただけで改善した証拠はない。

### アーキテクチャ⇄メトリクスの因果考察

固定weight8だけではこの条件のボトルネックを解消できない。動的勾配調整や長い更新数まで否定しない。
playerの悪化も同時に観測され、ballに重みを足すことを無条件の改善策としない。

### 既存実験との比較

直前のno-smooth weight1と同じ教師・seed・epoch・入力欠損設定を使った。
データ版も更新回数も変わる全体版61clipとの比較とは区別する。

### 次に有効な実験

weight8は採用せず、no-smooth weight1を全体版61clip・60epoch・1800更新で評価する。
testではなくvalidationの平均位置baseline差・motion・player精度で学習成立を判定する。
