---
id: run-plcs-reid-safe-mask-gpu-smoke-20260924
type: run
task: plcs
sequence: 120
recorded_at: '2026-09-24'
title: 全無効行対策後もcompiled validationが非有限となったGPU検証
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-24'
status: failed
config:
  model: plcs_player_reid
  compile: true
  batch_size: 4
  planned_epochs: 2
  sanity_validation_batches: 2
  hidden_dim: 256
  num_stages: 4
  data: plcs/tracked_person_reid_v1
metrics: {}
repro:
  commit: be54b85471c7e735544276f5038b2ab70b139798
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python tests/benchmarks/person_association.py
    --device cuda --data-root /home/kamimura/projects/tennis-lab/data --scene-dir
    plcs/tracked_person_reid_v1 --output /home/kamimura/projects/tennis-lab/outputs/plcs/diagnostics/tracked_reid_safe_mask_smoke_20260924
artifacts:
  run_dir: knowledge/runs/run-plcs-reid-safe-mask-gpu-smoke-20260924
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790219880496701332_779831_plcs-reid-safe-mask-gpu-smoke-20260924.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/diagnostics/tracked_reid_safe_mask_smoke_20260924
parents:
- run-plcs-reid-sdpa-probe-20260924
relations: []
papers: []
tags:
- reid
- gpu_smoke
- failed
- nonfinite
---

## 観測

無効attention行を自己参照させてから出力を0にする修正を適用し、CPUでは有効出力と勾配が従来maskと一致した。しかし異なる実データbatchを使うsanity validationで、sample index 4,5,6,7のlossがNaNになり、追加した非有限値guardが即時停止した。学習完走やtest精度として報告できる結果はない。TensorBoardには学習曲線を作れる更新記録がない。

単一batchのprobeでは修正後に反復成功したが、実データ全体へは一般化できなかった。副作用のある自動fallbackは導入せず、次のrunではconfig上でcompile=falseを明示する。maskの変更自体は有効queryから見えるkeyを変えないため保持する。checkpointの採用対象ではない。
