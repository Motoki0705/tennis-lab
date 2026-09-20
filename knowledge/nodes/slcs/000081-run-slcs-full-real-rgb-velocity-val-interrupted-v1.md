---
task: slcs
sequence: 81
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-full-real-rgb-velocity-val-interrupted-v1
type: run
title: 'SLCS教師速度整合val: Windows再起動後に9個の空出力を確認・採用不可'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball jerk=0, supervised ball velocity
  data: slcs/real_rgb_v1, recording-disjoint val
  input_modes: [full, no_rgb, detector_gap, rgb_only, detector_gap_no_rgb]
  selected_epoch_zero_based: 49
  test_executed: false
metrics: {}
repro:
  commit: c5b32165c27bba0319ff2328c19c73455ea67eda
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.evaluate_slcs_run --output-root
    /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_velocity/s42-001
    --output slcs/evaluate/real_rgb_velocity/s42-001 --domain-prefix video_=meiji
    --default-domain broadcast --device cuda --batch-size 4 --ball-train-mean --gap-no-rgb
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-velocity-val-interrupted-v1
  output_dir: outputs/slcs/evaluate/real_rgb_velocity/s42-001
  integrity_audit: knowledge/runs/run-slcs-full-real-rgb-velocity-val-interrupted-v1/integrity_audit.json
parents:
- run-slcs-full-real-rgb-velocity-e60-v1
relations:
- to: run-slcs-full-no-smooth-gap-rgb-val-v2
  rel: compares
tags: [slcs, real-rgb, velocity, validation, interrupted, artifact-integrity]
---

## 考察 / Findings

### 要約

queue台帳はdoneだが、Windows再起動後の出力を調べると32ファイル中9個が0バイトだった。
5条件の評価完了として採用できないため、本nodeはfailedとする。学習60epoch完了の親runとは区別する。
この事後点検は新たな推論・学習を実行せず、元出力も上書きしていない。

### アーキテクチャ詳細

train-only統計で固定した速度整合lossを持つモデルの、validation scene最良epoch49を公開評価CLIで選択した。
同じval splitに対してfull/no_rgb/detector_gap/rgb_only/detector_gap_no_rgbとtrain平均定数baselineを評価する計画だった。
選択結果は元出力のselection.jsonに残る。testは本コマンドの対象ではない。

### メトリクスの解釈

detector_gap_no_rgbの5ファイル、通常comparisonのJSON/CSV、gap_rgb_comparisonのJSON/CSVが空だった。
ファイルサイズ監査の件数はintegrity_audit.jsonを正本とし、モデル精度のmetricsには入れない。
他4条件には非空の保存物があるが、それだけで5条件の比較完了・採用可とは扱わない。
本runは学習曲線を持たず、親runの曲線を流用しない。

### アーキテクチャ⇄メトリクスの因果考察

Windowsの停止・再起動後に空ファイルを確認したという観測であり、特定SSD・RAM・ドライバーの故障や、
このloss変更がOS停止を引き起こしたという因果関係は未確定。
queueのプロセス完了状態と、再起動後の成果物の完全性は別に検証する必要がある。

### 既存実験との比較

基準の5条件評価は維持する。本runの欠損条件や比較表を別runから補完せず、不完全な出力を採否判断へ使わない。
速度整合の基準置換は保留であり、精度改善・退行が確定したrunとして集計しない。

### 次に有効な実験

ユーザーのWindows原因特定・再発防止の依頼を優先し、新しいGPU実験を保留する。
管理者確認が成立していないためダンプ解析は未実施で、GPU再投入より先にホストの診断を進める。
再開可能になったら元の不完全出力を保持し、別の一意な出力先へ同じ固定条件を再実行する。
全必須ファイルの検証後に位置誤差・裾・速度・visibility遷移・高速教師区間・playerを比較する。
