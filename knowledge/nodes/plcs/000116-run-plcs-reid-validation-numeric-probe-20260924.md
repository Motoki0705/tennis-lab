---
id: run-plcs-reid-validation-numeric-probe-20260924
type: run
task: plcs
sequence: 116
recorded_at: '2026-09-24'
title: fresh推論の数値切り分け
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-24'
status: done
config:
  batch_size: 4
  frames: 512
  hidden_dim: 256
  stages: 4
  checkpoint_epoch: 4
  checkpoint_role: numerical_diagnostic_only
metrics:
  finite_cases: 6
  case_count: 6
repro:
  commit: 53db1868054e8330126d34a458edb1424646d61e
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    .venv/bin/python /tmp/reid_numeric_probe.py
artifacts:
  run_dir: knowledge/runs/run-plcs-reid-validation-numeric-probe-20260924
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790219164436955028_758118_plcs-reid-validation-numeric-probe-20260924.log
parents:
- run-plcs-fixed-track-reid-e60-s42-20260924
relations: []
papers: []
tags:
- reid
- numeric_probe
- compile
- failed_hypothesis
---

## 観測と制限

CPU fp32、CUDA fp32/bf16、compiled bf16、no_grad/inference_modeの6条件で、保存epoch4重みと同一validation batchの出力・lossがすべて有限だった。モデル重み/入力の恒常的な非有限値という仮説を支持しない。単発forwardだけなので反復やmode切替の安定性は確認していない。

RTX 5060 Ti、共有training queue、PR #915のcommit 53db1868で実施した。入力と重みは前runから固定したもので、精度評価・追加学習・checkpoint採用の実験ではない。診断スクリプトと生ログをrun bundleへ保存した。元commandの`/tmp`スクリプトはbundleの`diagnostic.py`に対応する。TensorBoardを生成しない数値probeのため学習曲線はない。内部コンパイラの原因は未確定であり、既定の本学習は明示的にeagerへ変更する。
