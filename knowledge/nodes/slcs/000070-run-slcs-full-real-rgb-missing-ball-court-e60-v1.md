---
task: slcs
sequence: 70
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-full-real-rgb-missing-ball-court-e60-v1
type: run
title: 'SLCS欠損ballのcourt文脈保持: 60epoch・1800更新を完走'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4, missing_ball_court_context=true
  loss: no-ball-smooth, ball_velocity_weight=0
  data: slcs/real_rgb_v1, burst24
  max_epochs: 60
  seed: 42
  test_after_fit: false
metrics:
  completed_epochs: 60
  global_step: 1800
  terminal_train_ball_position_error_m: 2.452094316482544
  terminal_train_player_position_error_m: 1.31745445728302
  terminal_val_ball_position_error_m: 2.570133686065674
  terminal_val_player_position_error_m: 1.4388635158538818
  terminal_val_scene_position_error_m: 2.0044984817504883
  best_val_scene_position_error_m: 1.973695993423462
  selected_epoch_zero_based: 49
  context_weight_norm: 0.7447413802146912
  context_weight_nonzero: 5376
repro:
  commit: edabe593ef615d12a07a282a5df61e9c3c5a94d5
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_missing_ball_court paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    run.output_dir=slcs/train/real_rgb_missing_ball_court/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-missing-ball-court-e60-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_missing_ball_court/s42-001/logs/version_0
  curves: knowledge/runs/run-slcs-full-real-rgb-missing-ball-court-e60-v1/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_missing_ball_court/s42-001/logs/version_0
parents:
- run-slcs-full-real-rgb-no-ball-smooth-e60-v3
- run-slcs-full-real-rgb-velocity-val-v2
relations:
- to: run-slcs-full-real-rgb-velocity-e60-v1
  rel: compares
tags:
- slcs
- real-rgb
- missing-ball
- court-context
- architecture-ablation
---

## 考察 / Findings

### 要約

欠損ballのtokenへ観測済みcourt文脈を加える単独変更を、共有GPU queueのall予約で60epoch・1800更新実行した。
terminal checkpointのepoch59/global_step1800と全浮動小数重みの有限性、TensorBoard各60epochの系列を確認した。
validation scene最良はepoch49。testは実行せず、採否は別runの固定val5条件・境界診断で判断する。

### アーキテクチャ詳細

baselineと同じhidden128/shared4・DINO downsample2・burst24・seed42・lossを使い、ball jerkとvelocity lossはともに0。
ball不可視かつ非paddingのframeで、court UVとvalid flagsのbias無し線形射影を不可視ball tokenへ加算する。
無効court座標はmaskし、全court欠損・padding・ball観測ありには加算しない。
128×42の追加重みはRNGを進めずゼロ初期化し、初期の既存重み・出力を維持する。教師や隠されたball UVは入力しない。
設計動機のBRITS等との関係は実験groupに記録し、論文の再現や有効性の証明とは呼ばない。

### メトリクスの解釈

train ball 9.2091→2.4521m、player12.1132→1.3175m。terminal val ball2.5701m/player1.4389m。
最良val sceneは1.973696mで、終端2.004498mより小さい。追加射影の全5376重みが非ゼロとなり、Frobenius normは0.74474。
追加経路が最適化された証拠ではあるが、欠損境界の改善をこれだけでは示さない。
選定epoch49のSHA256は `3e59feda9f3f5ba1a52d85716c658f22546a8c8f3b9eff01aeb9cdbf4887f6a4`。
学習系列・選定候補とmonitorはsummary.jsonへ保存した。値は疑似教師との一致度であり独立実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察

観測済みcourt情報を欠損ballから利用できるようにしても、ballの時刻ごとの位置情報が直接増えるわけではない。
shared attentionやRGBからの復元も関与するため、curveやweight normだけで境界ジャンプの原因・解決を断定しない。
終盤にtyped configのPython既定値を除去する契約修正を行ったが、forward・helper・loss・学習dataは変更していない。
queue取得時のsource commitと解決済みconfigを再現根拠として保持する。

### 既存実験との比較

baselineのbest val scene1.956797mに対し1.973696mで小幅に悪い。速度整合候補の1.976942mとは近い。
ここでは終端training metricやmonitorだけで採用せず、同一選定手順の固定val5条件を比較する。
速度lossやburst長を同時に変更していない。既存baselineの自動test結果を係数・model選定へ利用していない。

### 次に有効な実験

選定epoch49をfull/no_rgb/detector_gap/rgb_only/detector_gap_no_rgbで評価する。
train-only高速閾値と既存paired transition CLIを維持し、両visibility境界、両端観測、高速教師区間、
位置平均・p95、Meiji/broadcast、playerを比較する。採用確定まではheld-out testを開かない。
