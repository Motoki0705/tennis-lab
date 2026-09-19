---
id: run-slcs-full-real-rgb-no-ball-smooth-e60-v3
type: run
title: 'SLCS実RGB全体版60epoch完走: 1800更新、validation ball約2.54m'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: SLCSLoss, ball_position_smoothness_weight=0
  data: slcs/real_rgb_v1, 61 clips
  max_epochs: 60
  optimizer_updates: 1800
  seed: 42
  augmentation_enabled: true
  precision: bf16-mixed
metrics:
  player_position_error_m: 1.215777
  player_position_error_median_m: 0.93444
  player_angular_error_deg: 33.348358
  player_angular_error_median_deg: 18.223864
  player_position_accuracy_0.3m: 0.052746
  player_position_accuracy_0.5m: 0.176837
  player_position_accuracy_1.0m: 0.54023
  player_position_accuracy_2.0m: 0.861909
  player_angle_accuracy_10deg: 0.335341
  player_angle_accuracy_15deg: 0.442626
  player_angle_accuracy_30deg: 0.642186
  player_position_pred_b_m: 0.51917
  player_rotation_pred_b_deg: 20.855762
  player_position_conf_error_corr: 0.376188
  player_rotation_conf_error_corr: 0.484354
  ball_position_error_m: 2.724171
  ball_position_error_median_m: 1.823494
  ball_position_accuracy_0.3m: 0.028068
  ball_position_accuracy_0.5m: 0.089849
  ball_position_accuracy_1.0m: 0.288122
  ball_position_accuracy_2.0m: 0.535244
  ball_position_pred_b_m: 0.812434
  ball_position_conf_error_corr: 0.294833
  scene_position_error_m: 1.969974
repro:
  commit: 61d94edb5c394de2cb7e905fce362da299bf73d9
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb loss.ball_position_smoothness_weight=0.0 run.output_dir=slcs/train/real_rgb_no_ball_smooth/s42-takeover-003
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-e60-v3
  predictions: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-e60-v3/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/logs/version_0
  log: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-e60-v3/queue.log
  curves: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-e60-v3/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/logs/version_0
parents:
- run-slcs-real-rgb-full-assembly-v1
- run-slcs-real-rgb-cpu-smoke-v1
- run-slcs-full-real-rgb-no-ball-smooth-interrupted-v2
relations:
- to: run-slcs-pilot-no-ball-smooth-e60-takeover-v1
  rel: compares
tags:
- slcs
- real-rgb
- full-dataset
- no-ball-smooth
- gpu
---

## 考察 / Findings

### 要約

Meiji 56clipとbroadcast 5clipの全体版で60epoch・1800更新を完走した。
共有queueのall予約で2026-09-19 12:39:02〜12:50:56 JSTに実行し、doneへ遷移した。
学習成立は前進したが、このrun単独ではRGBの寄与や欠損への頑健性を判定しない。

### アーキテクチャ詳細

`train_real_rgb`からball位置平滑化だけを0へ変更。ball位置weight=1、NLL=0.5を維持した。
固定seed42、batch16、LR3e-4、warmup200、window120、train/eval stride60/120、入力augmentationあり。
train/val/testは466/343/239窓で、収録・会場のsplitは交差しない。resolved_config.yamlが設定の正本。
先行2回の中断後にdatasetの反復setupを冪等にして不要な二重構築を除いたが、数値設定は変えていない。

### メトリクスの解釈

frontmatterとpred_test.npzはrunnerが自動保存した**終端last重みのtest**で、選定根拠ではない。
TensorBoardのtrain/val各60点の最終stepは1799。train ballは9.2116→2.5019m、playerは12.1232→1.2978m。
val終端はball2.5380m、player1.3886m。scene monitor最小1.956797mのepoch56を後続評価が選定する。
曲線は前半の減少と後半の頭打ちを示す。augmentation付きtrainと欠損なしvalの大小だけで過学習を判定しない。
すべて擬似教師との一致度であり、独立実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察

全体版ではtrain ballの低下が明確で、旧pilotの約6.48mに留まる状態から前進した。
データ量・教師版・総更新数・warmupが同時に異なるため、改善を一要因の効果とは断定しない。
今回CRCエラーや環境再起動は再発しなかったが、前回の原因解決を証明するものではない。

### 既存実験との比較

旧7clip pilotは360更新で、全体版は1800更新。validationの収録内clip数も異なるため、
旧pilotの7.0030mと全体版の2.5380mを同一評価集合での改善率として扱わない。
ball weight8は旧pilotで不採用となったため、このrunへ持ち込んでいない。

### 次に有効な実験

validation最良epoch56を固定し、同じ343窓のfull/no_rgb/detector_gap/rgb_onlyをfloat32で評価する。
train-only平均位置定数、実FPSの速度・位置分散、Meiji/broadcast別の誤差を併記する。
学習終端test値は施策やcheckpointの選定に使わない。
