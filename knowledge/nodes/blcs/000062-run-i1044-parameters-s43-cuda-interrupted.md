---
id: run-i1044-parameters-s43-cuda-interrupted
type: run
task: blcs
sequence: 62
recorded_at: '2026-10-14'
title: '候補Bの追加seed43初回: backward中のCUDA unknown errorで中断'
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: failed
config:
  arm: parameters
  seed: 43
  confirmation_seed: true
  requested_updates: 4000
  model: blcs_multiview_axial
  loss:
    position_weight: 1.0
    position_scale_m: 11.885
    position_beta_m: 11.885
    reprojection_weight: 0.1
    reprojection_beta_fraction: 1.0
    velocity_weight: 0.1
    velocity_scale_mps: 30.0
    velocity_beta_mps: 10.0
    rollout_position_weight: 1.0
    rollout_velocity_weight: 0.1
    rollout_horizons:
    - 1
    - 3
    physics_start_step: 200
    physics_warmup_steps: 400
    parameters_weight: 0.01
    event_weight: 0.01
    reconstruction_weight: 0.1
    parameter_consistency_weight: 0.1
    event_positive_weight: 10.0
  dataset: blcs/single_object-physics-v3-cc863cacd-20261007
  completed_comparison_run: false
  test_evaluated: false
metrics:
  last_logged_step: 2609
  last_saved_checkpoint_step: 2610
  last_logged_val_trajectory_rmse_m: 0.8086180686950684
repro:
  commit: a1921e349b1ec6162affccc6b96f38c4af2ad729
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv/bin/python -m src.tasks.blcs.experiments.scripts.repeat_selected --seed
    43 --data-root /home/kamimura/projects/tennis-lab/data
artifacts:
  run_dir: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted
  output_dir: outputs/blcs/i1044/aborted-parameters-s43-cuda-unknown
  log: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/queue-summary.log
  raw_log: {status: deleted, record: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/artifact-manifest.json#/files/0}
  log_archive: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/log-archive.json
  config: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/config.yaml
  experiment_config: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/experiment-config.yaml
  output_archive: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/output-archive.json
  interruption: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/interruption.json
  capture_checksums: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/capture_checksums.json
  scalars: {status: deleted, record: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/artifact-manifest.json#/files/1}
  tb_logdir: outputs/blcs/i1044/aborted-parameters-s43-cuda-unknown/logs/version_0
  curves: knowledge/runs/run-i1044-parameters-s43-cuda-interrupted/curves.png
parents:
- group-i1044-seed42-selection
relations:
- to: run-i1044-parameters-s42-u4000
  rel: replicates
papers: []
tags:
- issue1044
- interrupted
- confirmation-seed
- cuda-unknown
---

seed42のvalidationで固定した候補Bを追加seed43で4,000更新する予定だったが、backward中にCUDA unknown errorが表面化して停止し、optimizerをCPUへ戻すteardownでも同じエラーが生じた。driver末尾にはcuStreamSynchronizeの999、sticky error、cuStreamIsCapturingの999があるが、根本原因は未特定。既読のシステムログには対応するOOM・Xid・resetの証拠がなく、障害時刻のkernel記録にも欠落があるため、これらを原因から除外できない。完走runにも見られたINVALID_CONTEXT診断の反復だけで、損失実装やGPU全体の異常を断定しない。

最後のTensorBoard記録はstep 2609。これは最後に記録されたstepであり、正確な停止更新数とは限らない。最後のvalidationはlogger step 2474の軌道RMSE **0.808618069 m**で、正式な完了runの選定値としては扱わない。last.ckptに保存されたglobal_stepは2610。全途中checkpointのhash・更新数・epoch・保存時刻をinterruption.jsonへ記録し、重み本体とTensorBoardを退避先に保持した。未完了のためselection.jsonはなく、testは評価していない。

queueログ原本はgzipで無損失保存し、原本と展開後のSHA-256を照合した。読みやすいqueue-summary.logではCUDA driver診断行のみを明示して省略する。実行時commit・patch・commandの捕捉原本は変更していない。元のoutput_dir参照は履歴のまま残し、退避先との対応をoutput-archive.jsonに記録した。保存scalarから途中までの学習曲線を作り、未測定のtrain rollout診断は補間しない。

この中断試行は成功10 runに含めず、同じseed・学習条件・4,000更新を最初から再実行する。途中checkpointは引き継がない。新しいretryは既存FIFOの末尾へ投入し、未実行だった最終評価jobだけを取り消してretryより後へ再登録した。既存のB seed44・実行中の主対照seed44・他作業のjob順序は維持した。根本原因が特定されていない段階で科学的な設定やCUDA allocator設定を変えていない。
