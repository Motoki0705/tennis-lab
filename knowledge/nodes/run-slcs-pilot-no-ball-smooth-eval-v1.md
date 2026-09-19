---
id: run-slcs-pilot-no-ball-smooth-eval-v1
type: run
title: 'SLCS no-ball-smoothのvalidation評価: ball誤差7.00m、動きは増すが低分散が残る'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/real_rgb_pilot_v2
  selected_epoch_zero_based: 55
  split: val
  device: cpu
  precision: float32
metrics:
  val_windows: 43
  full_player_position_error_m: 2.6921656131744385
  full_ball_position_error_m: 7.003023147583008
  no_rgb_ball_position_error_m: 7.014247417449951
  detector_gap_ball_position_error_m: 7.007242202758789
  rgb_only_ball_position_error_m: 7.046573638916016
  full_meiji_ball_position_error_m: 7.37332820892334
  full_broadcast_ball_position_error_m: 5.827974319458008
  full_ball_predicted_mean_speed_mps: 0.9412690206486592
  full_ball_target_mean_speed_mps: 13.70674498997783
  full_ball_std_norm_ratio: 0.06986201838215896
repro:
  commit: 7621307bf292f5b06bd6d3fe2e0cf90a7ee27985
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python -m scripts.analysis.evaluate_slcs_run
    --output-root /home/kamimura/projects/tennis-lab/outputs
    --training-run slcs/train/real_rgb_pilot_no_ball_smooth/s42-takeover-001
    --output slcs/evaluate/real_rgb_pilot_no_ball_smooth/s42-takeover-001
    --domain-prefix video_=meiji --default-domain broadcast --device cpu --batch-size 4
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-eval-v1
  output_dir: outputs/slcs/evaluate/real_rgb_pilot_no_ball_smooth/s42-takeover-001
parents: [run-slcs-pilot-no-ball-smooth-e60-takeover-v1]
relations:
- {to: run-slcs-pilot-augmented-portable-eval-v1, rel: compares}
tags: [slcs, pilot, evaluation, motion, ball-collapse, cpu]
---

## 考察 / Findings

### 要約

validation最良epoch55を公開CLIで選び、val43窓の4条件比較を完了した。
ball full誤差はbaseline7.0308→7.0030mに留まり、十分な軌道学習・頑健性と判断しない。

### アーキテクチャ詳細

training保存configのdata/quality/window契約を維持し、augmentationだけ無効にしたCPU float32評価。
checkpoint SHAはc03165b52017c9a145dd5b4f807a746a5d5eda98979e92a908e0718e7d0ca580。
testは評価・選定とも行っていない。実FPS・連続有効frameのみでmotionを算出した。

### メトリクスの解釈

full/no_rgb/detector_gap/rgb_onlyのball誤差は7.0030/7.0142/7.0072/7.0466m。
full playerは2.6922m（baseline2.6691m）。ball予測平均速度0.9413m/sに対し教師13.7067m/s。
位置標準偏差ノルム比0.06986は以前の0.01831より大きいが、動くこと自体は正確さではない。
予測jerkも増え、速度誤差平均13.7944m/sと教師対応は乏しい。学習runではないため曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

平滑化を外すと出力の変化は増したが、低分散と大誤差は残った。
平滑化による抑制が一部あった可能性と、他の学習制約が残る可能性を区別する。
位置誤差改善0.4%程度を頑健な再構成の達成とは呼ばない。

### 既存実験との比較

baseline公開CLIと同じvalidation、mask、float32、入力条件を使う。
Meiji/broadcastごとの結果も保存し、43窓中33窓がMeijiという構成を隠さない。
旧train-mean実験の約7.0333mと近いが、新schemaに対する厳密な再比較は別処理で確認する。

### 次に有効な実験

同じno-smooth条件からball位置損失だけ8倍の60epoch比較へ進む。
GradNormの勾配配分という問題設定を参考にした固定係数の探索であり、動的GradNormの再現ではない。
根拠論文: https://proceedings.mlr.press/v80/chen18a.html 。効果の判断はvalidationに限定する。
