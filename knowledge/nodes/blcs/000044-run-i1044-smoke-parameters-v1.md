---
id: run-i1044-smoke-parameters-v1
type: run
task: blcs
sequence: 44
recorded_at: '2026-10-13'
title: '物理headスモーク初回: 短予算とwarmupの不整合'
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: failed
config:
  diagnostic: true
  completed_comparison_run: false
metrics: {}
repro:
  commit: 90e52a25126cb4d5a790e2a36ebfd09c9f1480da
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm parameters --seed 42 --steps 4 --smoke --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/smoke-parameters-v1
  complete: false
artifacts:
  run_dir: knowledge/runs/run-i1044-smoke-parameters-v1
  log: knowledge/runs/run-i1044-smoke-parameters-v1/queue.log
  output_dir: outputs/blcs/i1044/smoke-parameters-v1
  config_yaml: knowledge/runs/run-i1044-smoke-parameters-v1/config.yaml
  tb_logdir: outputs/blcs/i1044/smoke-parameters-v1/logs/version_0
  scalars: {status: deleted, record: knowledge/runs/run-i1044-smoke-parameters-v1/artifact-manifest.json#/files/0}
  original_capture: knowledge/runs/run-i1044-smoke-parameters-v1/original_capture
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

4更新のスモークに対してwarmupが長く、optimizer構築時の明示エラーで終了した。学習精度の証拠は得られていない。スモークではwarmupを短予算内に設定し、通常4,000更新の200更新warmupとは区別した。

初期queue captureは新規untrackedのexperiments/とphysics/をpatchに含めていなかった。捕捉原本を内容変更せずoriginal_capture/に保持したが、このスモークの厳密な再実行はできない。実行ログ・設定・保存結果に限った証拠として扱う。正式4,000更新runはこれらのファイルをcommitした版で実施する。
