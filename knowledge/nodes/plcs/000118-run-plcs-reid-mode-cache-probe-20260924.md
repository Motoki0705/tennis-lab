---
id: run-plcs-reid-mode-cache-probe-20260924
type: run
task: plcs
sequence: 118
recorded_at: '2026-09-24'
title: optimizer更新前にも現れるcompiled評価異常
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
metrics: {}
repro:
  commit: 53db1868054e8330126d34a458edb1424646d61e
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    .venv/bin/python /tmp/reid_transition_probe3.py
artifacts:
  run_dir: knowledge/runs/run-plcs-reid-mode-cache-probe-20260924
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790219436999325150_768381_plcs-reid-mode-cache-probe-20260924.log
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

compiledモデルでtrainモードのforwardだけを実行した後にもevalがNaNになった。backward/optimizer stepが必須の発生条件ではない。RoPE moduleのbufferは有限だった。ただし値の完全な同一性や内部kernelのworkspaceまでは検証していない。

RTX 5060 Ti、共有training queue、PR #915のcommit 53db1868で実施した。入力と重みは前runから固定したもので、精度評価・追加学習・checkpoint採用の実験ではない。診断スクリプトと生ログをrun bundleへ保存した。元commandの`/tmp`スクリプトはbundleの`diagnostic.py`に対応する。TensorBoardを生成しない数値probeのため学習曲線はない。内部コンパイラの原因は未確定であり、既定の本学習は明示的にeagerへ変更する。
