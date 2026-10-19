---
id: run-i1044-position-s42-cuda-interrupted
type: run
task: blcs
sequence: 52
recorded_at: '2026-10-13'
title: '位置のみ比較の初回実行: CUDA unknown errorで中断'
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
  commit: 83c96b0636dc28634a621b9c64e069de7793e4dd
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm position --seed 42 --steps 4000 --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/position-s42-u4000
artifacts:
  run_dir: knowledge/runs/run-i1044-position-s42-cuda-interrupted
  log: knowledge/runs/run-i1044-position-s42-cuda-interrupted/queue.log
  output_dir: outputs/blcs/i1044/aborted-position-s42-cuda-unknown
  experiment_config_yaml: knowledge/runs/run-i1044-position-s42-cuda-interrupted/experiment-config.yaml
  config_yaml: knowledge/runs/run-i1044-position-s42-cuda-interrupted/config.yaml
  interruption_json: knowledge/runs/run-i1044-position-s42-cuda-interrupted/interruption.json
  tb_logdir: outputs/blcs/i1044/aborted-position-s42-cuda-unknown/logs/version_0
  scalars: {status: deleted, record: knowledge/runs/run-i1044-position-s42-cuda-interrupted/artifact-manifest.json#/files/0}
  curves: knowledge/runs/run-i1044-position-s42-cuda-interrupted/curves.png
parents: []
relations: []
papers: []
tags:
- issue1044
- interrupted
---

4,000更新の途中でCUDA unknown errorが入力の有限値確認時に表面化し、optimizerの後片付けでも再発した。非同期CUDA実行のため元の故障箇所は特定できず、非有限lossやBLCSコードが原因という証拠もない。all予約中の失敗で、割当範囲外の利用やdriver/WSLの状態は確定できていない。同seed・同条件の最初からの再実行を登録し、この途中runは正式比較10件へ含めない。

保存scalarの最大stepは629。これは最後に記録されたstepで、停止した正確な更新数とは限らない。途中checkpointはローカルの退避出力先へ保持した。

再現資料のcommit・patch・commandはqueue捕捉原本を維持した。巨大checkpointはgitへ追加していない。
