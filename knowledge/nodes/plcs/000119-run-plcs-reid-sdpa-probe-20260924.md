---
id: run-plcs-reid-sdpa-probe-20260924
type: run
task: plcs
sequence: 119
recorded_at: '2026-09-24'
title: 全無効行maskとcompiled attentionの反証
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
  finite_cases: 9
  case_count: 22
repro:
  commit: 53db1868054e8330126d34a458edb1424646d61e
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    .venv/bin/python /tmp/reid_sdpa_probe.py
artifacts:
  run_dir: knowledge/runs/run-plcs-reid-sdpa-probe-20260924
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790219597183632120_772991_plcs-reid-sdpa-probe-20260924.log
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

同一batchのcompiled eval反復では初回だけ有限、以後NaNとなった。dropout=0でも解消しなかった。無効queryに自己参照だけを許して出力を捨てるmaskでは、同一batchでの反復とtrain/eval切替の6呼出しがすべて有限だった。math backend指定のvariantはNaNが残った。後続の異なるbatchを使うGPUスモークでは修正だけで解消しなかったため、この単一batch結果を一般化しない。

RTX 5060 Ti、共有training queue、PR #915のcommit 53db1868で実施した。入力と重みは前runから固定したもので、精度評価・追加学習・checkpoint採用の実験ではない。診断スクリプトと生ログをrun bundleへ保存した。元commandの`/tmp`スクリプトはbundleの`diagnostic.py`に対応する。TensorBoardを生成しない数値probeのため学習曲線はない。内部コンパイラの原因は未確定であり、既定の本学習は明示的にeagerへ変更する。
