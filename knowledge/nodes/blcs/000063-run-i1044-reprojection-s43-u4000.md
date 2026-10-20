---
id: run-i1044-reprojection-s43-u4000
type: run
task: blcs
sequence: 63
recorded_at: '2026-10-14'
title: '既存戦略に対応する位置＋clean再投影対照: seed43・4,000更新完了（全長test未評価）'
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  protocol: issue1044-v1
  arm: reprojection
  seed: 43
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
    velocity_weight: 0.0
    velocity_scale_mps: 30.0
    velocity_beta_mps: 10.0
    rollout_position_weight: 0.0
    rollout_velocity_weight: 0.0
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
  confirmation_seed: true
metrics:
  val_trajectory_rmse_m: 0.7631002068519592
repro:
  commit: 8faa9bd42da3483cb06e7288fcff9be7e9a54858
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv/bin/python -m src.tasks.blcs.experiments.scripts.issue1044 --arm reprojection
    --seed 43 --steps 4000 --data-root /home/kamimura/projects/tennis-lab/data --output
    blcs/i1044/reprojection-s43-u4000
artifacts:
  run_dir: knowledge/runs/run-i1044-reprojection-s43-u4000
  output_dir: outputs/blcs/i1044/reprojection-s43-u4000
  log: knowledge/runs/run-i1044-reprojection-s43-u4000/queue-summary.log
  raw_log: {status: deleted, record: knowledge/runs/run-i1044-reprojection-s43-u4000/artifact-manifest.json#/files/0}
  log_archive: knowledge/runs/run-i1044-reprojection-s43-u4000/log-archive.json
  experiment_config: knowledge/runs/run-i1044-reprojection-s43-u4000/experiment-config.yaml
  config: knowledge/runs/run-i1044-reprojection-s43-u4000/config.yaml
  selection: knowledge/runs/run-i1044-reprojection-s43-u4000/selection.json
  training_verification: knowledge/runs/run-i1044-reprojection-s43-u4000/training-verification.json
  capture_checksums: knowledge/runs/run-i1044-reprojection-s43-u4000/capture_checksums.json
  scalars: {status: deleted, record: knowledge/runs/run-i1044-reprojection-s43-u4000/artifact-manifest.json#/files/1}
  tb_logdir: outputs/blcs/i1044/reprojection-s43-u4000/logs/version_0
  checkpoint: /home/kamimura/artifacts/tennis-lab/blcs/progressive-losses-20261013/reprojection-s43-u4000/blcs-epoch=084-step=03825.ckpt
  checkpoint_storage: knowledge/runs/group-i1044-progressive-losses-u4000/checkpoint-storage.json
  relocated_selection: /home/kamimura/artifacts/tennis-lab/blcs/progressive-losses-20261013/reprojection-s43-u4000/selection.json
  checkpoint_sha256: 861df49ca556848a39b5f97ff0ea60c54e92c23c456bca1789c934db0c56e974
  curves: knowledge/runs/run-i1044-reprojection-s43-u4000/curves.png
parents:
- group-i1044-seed42-selection
relations:
- to: run-i1044-reprojection-s42-u4000
  rel: replicates
papers: []
tags:
- issue1044
- progressive-loss-ablation
- u4000
- validation-only
---

位置教師にclean再投影を加えた既存v3学習戦略の主対照。新方式と同じxyz/velocity head・データ・更新予算で比較する。速度headの教師とrollout・物理headの損失は加えない。 固定snapshot・axial base・xyz/velocity head共通の比較プロトコルに従い、seed43で4,000 optimizer updatesを完了した。学習完了のqueue状態・ログ・選定台帳とcheckpointのSHA-256を照合した。

validationの全100 scene（中心128frame、短いsceneは全長）で選んだcheckpointの軌道RMSEは **0.763100207 m**。checkpointのglobal_stepは3825、epoch値は84（0始まり）で、保存されたvalidation曲線の最小値と一致する。最終部分epochの明示validationも選定に含む。その他の記録された指標と損失/勾配ログはscalars.json.gzに保存した。指標別の支持数はこのTensorBoard記録に含まれず、最終保存評価で記録する。この値は選択に用いたvalidationの値であり、独立testの性能ではない。

train段階の診断器はrollout_horizons=()の設定なので、trainのrollout位置・速度指標は未測定（保存scalarではnull）である。rollout損失の計算とは別で、validationの両指標は計測している。学習曲線は主評価軸の位置・速度・再投影・rolloutと地面下率を示し、未測定点は線で埋めない。

学習処理の計測時間は9208.541秒。half予約で他runと資源を共有したため、所要時間から方式の計算効率の優劣は判断しない。学習開始時のcommit・patch・command・設定と原本照合用checksumを保存し、巨大checkpointはローカル出力に保持する。CLI移設前後の演算・解決済み設定の同値性は別の資格確認に記録している。再現コマンドは実行時の入口を維持し、後続変更に合わせて書き換えていない。

本runはseed42のvalidationで先に固定した主対照と候補Bの追加seed確認の一つで、方式の再選択には使わない。全10 runの完了後に全長test・物理残差・欠損/イベント/view数の層別・実観測診断を行う。本登録時点ではtest推論も採用判断も行っていない。validationの近接した値だけではrollout項の効果やseed間の再現性を断定できない。

seed42とはmodel初期化・学習seedを変え、validationのカメラ抽選seed42は固定した。同じ方式のseed42 runと設定をそろえ、seed間のばらつきと最終testでの効果を確認する。今回の選定値はvalidation上の値であり、確認seedの独立test結果はまだ得ていない。最終更新でのvalidation RMSEは0.769646049 mだった。 通常epoch checkpointのglobal_step 3825はlogger step 3824に対応し、checkpoint内の選定callbackのscoreと保存先を照合した。last.ckptが4,000更新に到達していることも確認した。

共通の速度headはこの主対照では教師を与えないため、位置差分の運動指標と区別する。

CUDA driver診断を含むログはgzipで無損失保存し、queue原本と展開後のSHA-256を照合した。queue-summary.logではdriver診断行のみを明示して省略する。
