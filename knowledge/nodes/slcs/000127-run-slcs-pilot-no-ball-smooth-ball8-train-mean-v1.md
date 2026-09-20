---
task: slcs
sequence: 127
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-pilot-no-ball-smooth-ball8-train-mean-v1
type: run
title: 'weight8のtrain-only平均比較: val定数より0.0177mだけ改善'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  fit_split: train
  split: val
  device: cpu
metrics:
  train_mean_error_m: 7.033323568923691
  full_model_error_m: 7.015640559296395
  full_model_minus_train_mean_error_m: -0.0176830096272953
repro:
  commit: e454473c28e5a591dc892586e8575e1b29daa6cb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python knowledge/runs/run-slcs-pilot-no-ball-smooth-train-mean-v2/probe.py
    --evaluation-root outputs/slcs/evaluate/real_rgb_pilot_no_ball_smooth_ball8/s42-takeover-001
    --output-dir outputs/slcs/analyze/pilot_no_ball_smooth_ball8_train_mean/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-ball8-train-mean-v1
  output_dir: outputs/slcs/analyze/pilot_no_ball_smooth_ball8_train_mean/s42-takeover-001
parents: [run-slcs-pilot-no-ball-smooth-ball8-eval-v1]
relations:
- {to: run-slcs-pilot-no-ball-smooth-train-mean-v2, rel: compares}
tags: [slcs, baseline, train-only, cpu]
---

## 考察 / Findings

### 要約

weight8モデルもval平均定数とほぼ同じ誤差。weight1よりbaseline差は小さく、採用根拠はない。

### アーキテクチャ詳細

比較先nodeと同じproduction train窓・品質重み・dedup規則のTrainBallMeanを使用。
valの4条件の保存済み配列をCPU後処理し、モデル再推論・学習は行っていない。

### メトリクスの解釈

full 7.015641m、train平均7.033324m、差-0.017683m。rgb_onlyは平均定数より0.027668m悪い。
全条件・domain・videoの比較を保存した。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

固定loss増幅が低分散出力を十分解消しなかったという別評価の結論と整合する。
baselineはtrainのみfitであり、valに最適化された定数と比べた結果ではない。

### 既存実験との比較

weight1との差は同じteacher/mask照合を通過した評価同士。微小な定数との差を実用精度と呼ばない。

### 次に有効な実験

全体版61clipの60epochでは更新数が1800に増えるため、同じ診断で学習成立を確認する。
