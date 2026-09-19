---
id: run-slcs-pilot-no-ball-smooth-train-mean-v2
type: run
title: '公開train-only平均baseline: no-smoothはval平均定数より0.0303mだけ改善'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  split: val
  fit_split: train
  device: cpu
metrics:
  train_windows: 88
  positive_weight_unique_camera_frames: 4919
  val_valid_window_occurrences: 4265
  train_mean_error_m: 7.033323568923691
  full_model_error_m: 7.003022836387012
  full_model_minus_train_mean_error_m: -0.030300732536678865
repro:
  commit: e454473c28e5a591dc892586e8575e1b29daa6cb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python knowledge/runs/run-slcs-pilot-no-ball-smooth-train-mean-v2/probe.py
    --evaluation-root outputs/slcs/evaluate/real_rgb_pilot_no_ball_smooth/s42-takeover-001
    --output-dir outputs/slcs/analyze/pilot_no_ball_smooth_train_mean/s42-takeover-002
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-no-ball-smooth-train-mean-v2
  output_dir: outputs/slcs/analyze/pilot_no_ball_smooth_train_mean/s42-takeover-002
parents: [run-slcs-pilot-no-ball-smooth-eval-v1, run-slcs-pilot-no-ball-smooth-legacy-mean-failure-v1]
relations:
- {to: run-slcs-ball-train-mean-v1, rel: confirms}
tags: [slcs, baseline, train-only, cpu]
---

## 考察 / Findings

### 要約

新しいTrainBallMean APIで保存4条件NPZを再推論なしに評価した。fitはtrainのみ。
fullモデルは平均位置定数より0.0303m（約0.43%）良いだけで、十分な軌道学習とはいえない。

### アーキテクチャ詳細

production label-only窓の同一(video,clip,camera,frame)を整合性確認して重複除去し、confidence重み付き算術平均をfit。
平均は[-0.480405,0.404969,1.813897]m。val/test値をfitに使わず、DINO cacheは読まない。
評価は既存headlineと同じ非重み付き有効frame出現平均で、窓重複は維持する。
添付probe.pyは新APIの単純な呼出しで、repro.commit時点のAPIを使った実行sourceを同梱した。

### メトリクスの解釈

train 88窓・正weight重複除去後4919 camera-frame、val4265 frame出現。
平均baseline 7.033324mは旧実験結果と一致し、新schemaでも同じ比較を再現した。
4入力条件・domain/video別を保存。算術平均は平均Euclidean距離を最小化する定数ではない。曲線対象外。

### アーキテクチャ⇄メトリクスの因果考察

位置誤差が定数と近いという診断を数値で確認したが、それだけで入力を完全無視すると断定しない。
別runのmotion/感度と合わせた学習不足の証拠として扱う。

### 既存実験との比較

旧probeのschema非互換を隠さず別失敗nodeに保持し、新しい型付きAPIではteacher/windowの完全照合を通過した。
親で38対象tests、実装側でSLCS174 testsを確認済み。

### 次に有効な実験

全体版の選定重みにも同じtrain-only baselineを計算し、入力欠損劣化だけを成功基準にしない。
