---
id: run-i1044-reprojection-s42-cuda-interrupted
type: run
task: blcs
sequence: 54
recorded_at: '2026-10-13'
title: '既存戦略の対照初回: CUDA unknown errorで中断'
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: failed
config:
  arm: reprojection
  seed: 42
  requested_updates: 4000
  completed_comparison_run: false
metrics: {}
repro:
  commit: e8d7f72d507fcd398fd6e8bade5c2d275e7f8e4f
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm reprojection --seed 42 --steps 4000 --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/reprojection-s42-u4000
artifacts:
  run_dir: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted
  log: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted/queue.log
  output_dir: outputs/blcs/i1044/aborted-reprojection-s42-cuda-unknown
  experiment_config_yaml: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted/experiment-config.yaml
  config_yaml: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted/config.yaml
  interruption_json: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted/interruption.json
  scalars: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted/scalars.json.gz
  tb_logdir: outputs/blcs/i1044/aborted-reprojection-s42-cuda-unknown/logs/version_0
  curves: knowledge/runs/run-i1044-reprojection-s42-cuda-interrupted/curves.png
parents:
- run-i1044-position-s42-cuda-interrupted
relations: []
papers: []
tags:
- issue1044
- interrupted
---

位置＋再投影の既存戦略対照を4,000更新の予定で開始したが、CUDA unknown errorで途中停止した。入力の有限値検査でエラーが表面化し、optimizerの後片付け時にも再発した。非同期CUDAのため元の原因は未特定。並行していたvelocity条件は継続しており、このログだけからGPU全体のresetや損失実装の不具合とは断定できない。

保存scalarの最大stepは899。これは最後に記録されたstepであり、正確な停止更新数とは限らない。途中checkpoint・TensorBoardは退避出力先に保持し、queue捕捉原本も変更していない。正式比較10件には含めず、同じseed・学習条件で最初から再試行する。再試行にはCUDA_LOG_FILE=stderrを追加してruntime診断を残す。

未完了runなのでtest評価・採否判断は行わない。学習曲線は途中までの診断に限る。
