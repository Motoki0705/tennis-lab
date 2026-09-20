---
task: slcs
sequence: 54
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-rgb-pilot-gpu-smoke-v3
type: run
title: 実RGB SLCSの通常学習・保存経路の1epoch確認
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: slcs_fusion_hidden128_layers4
  loss: default
  data: real_rgb_pilot_v2
  epochs: 1
  checkpoint_monitor: val/scene_position_error_m_epoch
metrics:
  player_position_error_m: 5.166308
  player_position_error_median_m: 4.702977
  player_angular_error_deg: 143.719604
  player_angular_error_median_deg: 147.985626
  player_position_accuracy_0.3m: 0.0
  player_position_accuracy_0.5m: 0.0
  player_position_accuracy_1.0m: 0.0
  player_position_accuracy_2.0m: 0.0
  player_angle_accuracy_10deg: 0.0
  player_angle_accuracy_15deg: 0.0
  player_angle_accuracy_30deg: 0.0
  player_position_pred_b_m: 4.543619
  player_rotation_pred_b_deg: 75.304512
  player_position_conf_error_corr: 0.603603
  player_rotation_conf_error_corr: -0.609048
  ball_position_error_m: 6.16033
  ball_position_error_median_m: 5.93583
  ball_position_accuracy_0.3m: 0.0
  ball_position_accuracy_0.5m: 0.0
  ball_position_accuracy_1.0m: 0.023952
  ball_position_accuracy_2.0m: 0.08982
  ball_position_pred_b_m: 5.256995
  ball_position_conf_error_corr: -0.561479
  scene_position_error_m: 5.663319
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
    .venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot
    training.trainer.max_epochs=1 training.warmup_steps=0 run.output_dir=slcs/train/real_rgb_pilot_gpu_smoke/s42-003
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v3
  predictions: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v3/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_gpu_smoke/s42-003/logs/version_0
  curves: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v3/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_pilot_gpu_smoke/s42-003/logs/version_0
parents:
- run-slcs-rgb-pilot-gpu-smoke-v2
relations: []
tags:
- slcs
- real-rgb
- smoke
- checkpoint
---

## 考察 / Findings

### 要約
実RGBの7クリップで通常の1epoch学習、validation、TensorBoard、最良checkpointとlast保存、test予測の保存が完了した。60epoch精度比較へ進めるための動作確認である。

### アーキテクチャ詳細
train_real_rgb_pilotの128幅・4共有層SLCSとDINOv3特徴を使用。実験版real_rgb_pilot_v2はMeijiのtrain/val各1クリップと品質確認したbroadcast5クリップ。confidenceを明示変換した新しいMeiji教師へ更新した。max_epochs=1、warmup_steps=0で実行し、checkpointはepoch全体のscene_position_error_mを監視する。

### メトリクスの解釈
testはbroadcast Eastbourneのみで、1epoch終了時の人物位置5.166308m、ボール位置6.160330m、scene平均5.663319m。学習初期の動作確認値であり、目標精度に達した結果ではない。Meiji test収録は含まない。参照は擬似教師で実測3D正解ではない。

### アーキテクチャ⇄メトリクスの因果考察
人物角度143.72度などの大きな誤差は実際に観測されたが、1epochだけでは原因や収束後の性能を判断できない。今回確かめたのは、実データの入力検証と最良重み選定・再現bundle出力までの実行契約である。

### 既存実験との比較
直前のv2は存在しないmonitor名により保存時に停止した。val/scene_position_error_m_epochへ設定を合わせることで通常経路が完了した。小型CPU fixtureでも本物profile/runner、logger、checkpointを有効にした回帰テストを追加した。

### 次に有効な実験
同じ7クリップ・seed42・60epochで通常学習と入力欠損augmentationを比較し、選定した同一checkpointについてfull/no_rgb/detector_gap/rgb_onlyを評価する。全体版のMeiji testを含む評価を後続とする。
