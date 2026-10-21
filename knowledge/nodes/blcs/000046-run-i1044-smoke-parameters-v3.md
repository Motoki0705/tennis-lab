---
id: run-i1044-smoke-parameters-v3
type: run
task: blcs
sequence: 46
recorded_at: '2026-10-13'
title: 物理head付き通常経路の4更新スモーク
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  diagnostic: true
  completed_comparison_run: false
  arm: parameters
  seed: 42
  optimizer_updates: 4
  smoke: true
metrics:
  val_trajectory_rmse_m: 9.313501358032227
repro:
  commit: 90e52a25126cb4d5a790e2a36ebfd09c9f1480da
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm parameters --seed 42 --steps 4 --smoke --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/smoke-parameters-v3
  complete: false
artifacts:
  run_dir: knowledge/runs/run-i1044-smoke-parameters-v3
  log: knowledge/runs/run-i1044-smoke-parameters-v3/queue.log
  output_dir: outputs/blcs/i1044/smoke-parameters-v3
  experiment_config_yaml: knowledge/runs/run-i1044-smoke-parameters-v3/experiment-config.yaml
  config_yaml: knowledge/runs/run-i1044-smoke-parameters-v3/config.yaml
  selection_json: knowledge/runs/run-i1044-smoke-parameters-v3/selection.json
  tb_logdir: outputs/blcs/i1044/smoke-parameters-v3/logs/version_0
  scalars: {status: deleted, record: knowledge/runs/run-i1044-smoke-parameters-v3/artifact-manifest.json#/files/0}
  original_capture: knowledge/runs/run-i1044-smoke-parameters-v3/original_capture
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

物理headを含む通常学習・最終validation・checkpoint保存まで4更新で完了した。validation値はごく短いスモークの処理確認であり、方式Bの品質や優位性を示さない。testは使っていない。

選定validationの軌道RMSEは 9.313501 m、処理時間は 66.56秒。選定にtestは使用していない。

初期queue captureは新規untrackedのexperiments/とphysics/をpatchに含めていなかった。捕捉原本を内容変更せずoriginal_capture/に保持したが、このスモークの厳密な再実行はできない。実行ログ・設定・保存結果に限った証拠として扱う。正式4,000更新runはこれらのファイルをcommitした版で実施する。

kg_curvesで確認したが、4更新以下のスモークを収束曲線として解釈できないため、図は掲載せず全scalarを保存した。
