---
id: run-slcs-pilot-no-ball-smooth-ball8-eval-v1
type: run
title: 'ball位置weight8のvalidation評価: ball改善なし・player悪化のため不採用'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  split: val
  selected_epoch_zero_based: 56
  device: cpu
  precision: float32
metrics:
  val_windows: 43
  full_ball_position_error_m: 7.015640735626221
  full_player_position_error_m: 2.8882179260253906
  no_rgb_ball_position_error_m: 7.023316860198975
  detector_gap_ball_position_error_m: 7.016808032989502
  rgb_only_ball_position_error_m: 7.0609917640686035
  full_ball_predicted_mean_speed_mps: 1.2164152817459752
  full_ball_target_mean_speed_mps: 13.70674498997783
  full_ball_std_norm_ratio: 0.08281839317553723
repro:
  commit: 7621307bf292f5b06bd6d3fe2e0cf90a7ee27985
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python -m scripts.analysis.evaluate_slcs_run
    --output-root /home/kamimura/projects/tennis-lab/outputs
    --training-run slcs/train/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001
    --output slcs/evaluate/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001
    --domain-prefix video_=meiji --default-domain broadcast --device cpu --batch-size 4
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-ball8-eval-v1
  output_dir: outputs/slcs/evaluate/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001
parents: [run-slcs-pilot-no-ball-smooth-ball8-e60-v1]
relations:
- {to: run-slcs-pilot-no-ball-smooth-eval-v1, rel: compares}
tags: [slcs, evaluation, pilot, loss-weight, rejected]
---

## 考察 / Findings

### 要約

validation選定epoch56の4条件CPU評価は完了。weight1に対しball7.0030→7.0156m、player2.6922→2.8882mと悪化した。
本条件のweight8は不採用。testは選定・本評価とも用いない。

### アーキテクチャ詳細

同じ公開CLI・同じquality/window/teacher/maskを使用。checkpoint SHAは
b3a89dcdb2a549b470d360ad44924b4cc23fe48e4371051bcc1d8b235b4a03da。
full/no_rgb/detector_gap/rgb_onlyを対比較し、実FPSでmotionを算出した。

### メトリクスの解釈

位置分散比0.08282・予測速度1.2164m/sはweight1より増えたが、速度誤差も13.9839m/sへ増えた。
教師平均速度13.7067m/sに対して依然弱い。単に出力が動くことを軌道精度や頑健性とは扱わない。
新規学習runではないため収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

ball位置勾配を強めるだけでは360更新のpilotで十分な対応を学べなかった。
この結果は固定係数8の局所的否定であり、勾配調整一般や他の学習条件の否定ではない。

### 既存実験との比較

親のno-smooth weight1評価と同じval43窓、4265有効ball frame出現。
player悪化をballとの合成値だけで隠さず、4条件全てを保存した。

### 次に有効な実験

weight1に戻して全61clip版を60epoch学習し、固定train平均との比較も併記する。
