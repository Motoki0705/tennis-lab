---
id: run-slcs-pilot-no-ball-smooth-legacy-mean-failure-v1
type: run
title: '旧train-mean probeと公開evaluation schemaの非互換で停止'
provider: codex
date: '2026-09-19'
status: failed
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/real_rgb_pilot_v2
  device: cpu
metrics:
  completed_comparisons: 0
  exit_code: 1
repro:
  commit: 7621307bf292f5b06bd6d3fe2e0cf90a7ee27985
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-ball-train-mean-v1/probe.py
    --training-config outputs/slcs/train/real_rgb_pilot_no_ball_smooth/s42-takeover-001/config.yaml
    --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_no_ball_smooth/s42-takeover-001/val/full
    --output-dir outputs/slcs/analyze/pilot_no_ball_smooth_train_mean/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-legacy-mean-failure-v1
  output_dir: outputs/slcs/analyze/pilot_no_ball_smooth_train_mean/s42-takeover-001
parents: [run-slcs-pilot-no-ball-smooth-eval-v1, run-slcs-ball-train-mean-v1]
tags: [slcs, cpu, failed, evaluation-schema]
---

## 考察 / Findings

### 要約

旧実験専用probeで新しい公開CLIの結果を読もうとし、`KeyError: dataset_root`で停止した。
新schemaには旧probeが想定するcontext.dataset_rootがなく、モデルやdatasetの失敗ではない。

### アーキテクチャ詳細

archival probeは変更せず、training configとval/full保存先だけ指定した。
train-only fit後のevaluation context照合で停止し、評価値は公開していない。

### メトリクスの解釈

比較完了0、終了コード1。status.jsonに実行時の入力digestと例外を保存した。
モデル再推論や新しい学習はなく、曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

実験専用schemaと公開evaluation schemaを同一とみなした呼出しの誤り。
欠落キーを推測で補うfallbackは入れず、現schemaから型付き設定を構成する汎用baselineへ移行する。

### 既存実験との比較

元probeの過去runは元schemaで成功しており、今回の非互換は過去結果を無効にしない。
新しい4条件モデル評価も別runとして正常完了している。

### 次に有効な実験

開発中のTrainBallMean APIで既存NPZから再推論なしに比較し、新しい出力先に保存する。
