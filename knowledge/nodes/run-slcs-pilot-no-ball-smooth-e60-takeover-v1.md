---
id: run-slcs-pilot-no-ball-smooth-e60-takeover-v1
type: run
title: 'SLCS pilot 60epoch: ball平滑化のみ0、train ball誤差は小幅改善'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: SLCSLoss, ball_position_smoothness_weight=0 only
  data: slcs/real_rgb_pilot_v2, 7 clips
  max_epochs: 60
  seed: 42
  augmentation_enabled: true
metrics:
  player_position_error_m: 2.856937
  player_position_error_median_m: 1.886475
  player_angular_error_deg: 47.853481
  player_angular_error_median_deg: 37.64616
  player_position_accuracy_0.3m: 0.0
  player_position_accuracy_0.5m: 0.015982
  player_position_accuracy_1.0m: 0.196347
  player_position_accuracy_2.0m: 0.531963
  player_angle_accuracy_10deg: 0.054795
  player_angle_accuracy_15deg: 0.082192
  player_angle_accuracy_30deg: 0.22831
  player_position_pred_b_m: 1.069071
  player_rotation_pred_b_deg: 23.13028
  player_position_conf_error_corr: 0.459876
  player_rotation_conf_error_corr: -0.290837
  ball_position_error_m: 4.931725
  ball_position_error_median_m: 4.378894
  ball_position_accuracy_0.3m: 0.0
  ball_position_accuracy_0.5m: 0.011976
  ball_position_accuracy_1.0m: 0.041916
  ball_position_accuracy_2.0m: 0.191617
  ball_position_pred_b_m: 2.431465
  ball_position_conf_error_corr: 0.590216
  scene_position_error_m: 3.894331
repro:
  commit: 7621307bf292f5b06bd6d3fe2e0cf90a7ee27985
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot_no_ball_smooth
    run.output_dir=slcs/train/real_rgb_pilot_no_ball_smooth/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-e60-takeover-v1
  predictions: knowledge/runs/run-slcs-pilot-no-ball-smooth-e60-takeover-v1/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_no_ball_smooth/s42-takeover-001/logs/version_0
  curves: knowledge/runs/run-slcs-pilot-no-ball-smooth-e60-takeover-v1/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_pilot_no_ball_smooth/s42-takeover-001/logs/version_0
parents: [run-slcs-rgb-pilot-augmented-e60-v2, run-slcs-ball-gradient-probe-v1]
relations:
- {to: run-slcs-rgb-pilot-augmented-e60-v2, rel: compares}
tags: [slcs, pilot, real-rgb, ball-smoothness, single-intervention]
---

## 考察 / Findings

### 要約

旧7clip pilotのseed・60epoch・augmentationを固定し、ball位置平滑化のweightだけ1→0として完走した。
終端train ball誤差は旧6.586mから6.476mへ小幅改善したが、学習成立と呼べる誤差ではない。
新しい61clip全体版の学習ではない。

### アーキテクチャ詳細

`train_real_rgb_pilot_no_ball_smooth`を使用。ballの位置Smooth L1 weight=1、NLL weight=0.5を維持し、
player、model、data、seed42、batch16、warmup20、LR3e-4はbaselineと同じ。
GPU0を共有queueのall予約で使用した。実行時設定をresolved_config.yamlへ保存した。

### メトリクスの解釈

frontmatterとpred_test.npzはrunnerが自動出力した**終端last重みのtest**であり、モデル選定に使わない。
主な選定根拠は別nodeのvalidation最良重みの4条件比較とする。
TensorBoardの60点ではtrain playerが13.7694→2.0276m、ballが8.9864→6.4757m。
val終端はplayer2.7429m、ball7.0035m。scene monitor最小4.848259mを記録したepoch55を後続評価で選ぶ。
curves.pngはplayer学習とball頭打ちが残る様子を示す。擬似教師一致度であり実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察

平滑化を除いてもtrain ball誤差が大きく、平滑化だけが唯一の原因という仮説を支持しない。
学習が完全に停止していたわけではなく、勾配配分・入力融合・360更新という短い探索等が候補として残る。

### 既存実験との比較

augmented baselineとの単一loss比較。旧pilotの教師版とwindow数を維持し、全体版への変更と混同しない。
OS thread設定などの実行環境差はcommandに明示し、単一seedの結果から一般化しない。
既存の入力感度診断と併せ、入力欠損時の小さい誤差増加だけで頑健性を主張しない。

### 次に有効な実験

同じno-smooth条件からball_position_weightだけ1→8へ変更した60epochを比較する。
これは既存のplayer優位な共有勾配診断を踏まえた固定weightの仮説検証であり、GradNorm自体の実装ではない。
test値を条件選定に使わず、validation位置誤差・定数baseline・motionとplayer精度で判断する。
