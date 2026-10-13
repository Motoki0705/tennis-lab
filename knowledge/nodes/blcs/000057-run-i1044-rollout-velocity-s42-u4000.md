---
id: run-i1044-rollout-velocity-s42-u4000
type: run
task: blcs
sequence: 57
recorded_at: '2026-10-13'
title: '短時間rollout速度整合追加（候補A）: seed42・4,000更新完了（全長test未評価）'
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  protocol: issue1044-v1
  arm: rollout_velocity
  seed: 42
  optimizer_updates: 4000
  model: blcs_multiview_axial
  predict_velocity: true
  physics_head: false
  dataset: blcs/single_object-physics-v3-cc863cacd-20261007
  batch_size: 2
  accumulate_grad_batches: 8
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
    parameters_weight: 0.0
    event_weight: 0.0
    reconstruction_weight: 0.0
    parameter_consistency_weight: 0.0
    event_positive_weight: 10.0
  selection_used_test: false
  test_evaluated: false
  completed_comparison_run: true
metrics:
  val_trajectory_rmse_m: 0.6923530697822571
repro:
  commit: 806b0d3675e8bcd2e0af78006cf75a178f8b8ddf
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv/bin/python -m src.tasks.blcs.experiments.scripts.issue1044 --arm rollout_velocity
    --seed 42 --steps 4000 --data-root /home/kamimura/projects/tennis-lab/data --output
    blcs/i1044/rollout_velocity-s42-u4000
artifacts:
  run_dir: knowledge/runs/run-i1044-rollout-velocity-s42-u4000
  output_dir: outputs/blcs/i1044/rollout_velocity-s42-u4000
  log: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/queue-summary.log
  experiment_config: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/experiment-config.yaml
  config: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/config.yaml
  selection: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/selection.json
  training_verification: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/training-verification.json
  capture_checksums: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/capture_checksums.json
  scalars: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/scalars.json.gz
  tb_logdir: outputs/blcs/i1044/rollout_velocity-s42-u4000/logs/version_0
  checkpoint: /home/kamimura/projects/tennis-lab/.claude/worktrees/issue-1044-blcs-physics/outputs/blcs/i1044/rollout_velocity-s42-u4000/checkpoints/final-selected.ckpt
  checkpoint_sha256: 0a623d7328a45049332f33e28f603423ab1acd6d15d4a5caef8de01ab372d833
  raw_log: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/queue.log.gz
  log_archive: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/log-archive.json
  curves: knowledge/runs/run-i1044-rollout-velocity-s42-u4000/curves.png
parents:
- group-i1044-qualification
relations:
- to: run-i1044-rollout-position-s42-u4000
  rel: compares
papers: []
tags:
- issue1044
- progressive-loss-ablation
- u4000
- validation-only
---

rollout位置整合までの条件に、同じ共有積分器の短時間rollout速度整合を加えた候補A。rollout_position条件との差がこの項の追加効果を調べる。 固定snapshot・axial base・xyz/velocity head共通の比較プロトコルに従い、seed42で4,000 optimizer updatesを完了した。学習完了のqueue状態・ログ・選定台帳とcheckpointのSHA-256を照合した。

validationの全100 scene（中心128frame、短いsceneは全長）で選んだcheckpointの軌道RMSEは **0.692353070 m**。checkpointのglobal_stepは4000、epoch値は89（0始まり）で、保存されたvalidation曲線の最小値と一致する。最終部分epochの明示validationも選定に含む。その他の指標・支持数と損失/勾配ログはscalars.json.gzに保存した。この値は選択に用いたvalidationの値であり、独立testの性能ではない。

train段階の診断器はrollout_horizons=()の設定なので、trainのrollout位置・速度指標は未測定（保存scalarではnull）である。rollout損失の計算とは別で、validationの両指標は計測している。学習曲線は主評価軸の位置・速度・再投影・rolloutと地面下率を示し、未測定点は線で埋めない。

学習処理の計測時間は13802.951秒。half予約で他runと資源を共有したため、所要時間から方式の計算効率の優劣は判断しない。学習開始時のcommit・patch・command・設定と原本照合用checksumを保存し、巨大checkpointはローカル出力に保持する。CLI移設前後の演算・解決済み設定の同値性は別の資格確認に記録している。再現コマンドは実行時の入口を維持し、後続変更に合わせて書き換えていない。

同一seedの比較対照と残りの段階条件は学習中または待機中である。全6条件のvalidationが揃ってから追加seed対象を選び、計10 runの後に全長test・物理残差・欠損/イベント/view数の層別・実観測診断を行う。本登録時点ではtest推論も採用判断も行っていない。validationの近接した値だけではrollout項の効果やseed間の再現性を断定できない。

CUDA_LOG_FILE=stderrで得た診断ログには、cuCtxGetDevice_v2のCUDA_ERROR_INVALID_CONTEXTと現在のCUDA contextがないという行がそれぞれ2,475,203件あり、開始時から終了後まで出力された。max_steps到達・保存・選定台帳作成まで完了し、ログのPython例外・OOM・非有限値の検索ヒットは0だった。この完走runでも同じ診断が出るため、それだけで以前のCUDA unknown停止の原因とは断定できない。元の約487.6MBのログはqueueに保持し、全量を約2.4MBのgzipとして保存した。展開後のSHA-256を元ログと照合し、閲覧用のqueue-summary.logはCUDA driver診断行だけを省略したと明記している。
