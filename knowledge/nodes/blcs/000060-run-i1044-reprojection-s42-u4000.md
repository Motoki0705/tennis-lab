---
id: run-i1044-reprojection-s42-u4000
type: run
task: blcs
sequence: 60
recorded_at: '2026-10-13'
title: '既存戦略に対応する位置＋clean再投影対照: seed42・4,000更新完了（全長test未評価）'
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  protocol: issue1044-v1
  arm: reprojection
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
metrics:
  val_trajectory_rmse_m: 0.8476958870887756
repro:
  commit: 6273dbb114f2ad83c478ddd617ec8513c0843c08
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv/bin/python -m src.tasks.blcs.experiments.scripts.issue1044 --arm reprojection
    --seed 42 --steps 4000 --data-root /home/kamimura/projects/tennis-lab/data --output
    blcs/i1044/reprojection-s42-u4000
artifacts:
  run_dir: knowledge/runs/run-i1044-reprojection-s42-u4000
  output_dir: outputs/blcs/i1044/reprojection-s42-u4000
  log: knowledge/runs/run-i1044-reprojection-s42-u4000/queue-summary.log
  raw_log: {status: deleted, record: knowledge/runs/run-i1044-reprojection-s42-u4000/artifact-manifest.json#/files/0}
  log_archive: knowledge/runs/run-i1044-reprojection-s42-u4000/log-archive.json
  experiment_config: knowledge/runs/run-i1044-reprojection-s42-u4000/experiment-config.yaml
  config: knowledge/runs/run-i1044-reprojection-s42-u4000/config.yaml
  selection: knowledge/runs/run-i1044-reprojection-s42-u4000/selection.json
  training_verification: knowledge/runs/run-i1044-reprojection-s42-u4000/training-verification.json
  capture_checksums: knowledge/runs/run-i1044-reprojection-s42-u4000/capture_checksums.json
  scalars: {status: deleted, record: knowledge/runs/run-i1044-reprojection-s42-u4000/artifact-manifest.json#/files/1}
  tb_logdir: outputs/blcs/i1044/reprojection-s42-u4000/logs/version_0
  checkpoint: /home/kamimura/artifacts/tennis-lab/blcs/progressive-losses-20261013/reprojection-s42-u4000/final-selected.ckpt
  checkpoint_storage: knowledge/runs/group-i1044-progressive-losses-u4000/checkpoint-storage.json
  relocated_selection: /home/kamimura/artifacts/tennis-lab/blcs/progressive-losses-20261013/reprojection-s42-u4000/selection.json
  checkpoint_sha256: 97e88f588aa7ea57ec7f07b450d1c9c6c609a049b5a6f4a8e75b818d09667d01
  curves: knowledge/runs/run-i1044-reprojection-s42-u4000/curves.png
parents:
- group-i1044-qualification
relations:
- to: run-i1044-position-s42-u4000
  rel: compares
- to: run-i1044-reprojection-s42-cuda-interrupted
  rel: reruns
papers: []
tags:
- issue1044
- progressive-loss-ablation
- u4000
- validation-only
---

位置教師にclean再投影を加えた既存v3学習戦略の主対照。新方式と同じxyz/velocity head・データ・更新予算で比較する。速度headの教師とrollout・物理headの損失は加えない。 固定snapshot・axial base・xyz/velocity head共通の比較プロトコルに従い、seed42で4,000 optimizer updatesを完了した。学習完了のqueue状態・ログ・選定台帳とcheckpointのSHA-256を照合した。

validationの全100 scene（中心128frame、短いsceneは全長）で選んだcheckpointの軌道RMSEは **0.847695887 m**。checkpointのglobal_stepは4000、epoch値は89（0始まり）で、保存されたvalidation曲線の最小値と一致する。最終部分epochの明示validationも選定に含む。その他の記録された指標と損失/勾配ログはscalars.json.gzに保存した。指標別の支持数はこのTensorBoard記録に含まれず、最終保存評価で記録する。この値は選択に用いたvalidationの値であり、独立testの性能ではない。

train段階の診断器はrollout_horizons=()の設定なので、trainのrollout位置・速度指標は未測定（保存scalarではnull）である。rollout損失の計算とは別で、validationの両指標は計測している。学習曲線は主評価軸の位置・速度・再投影・rolloutと地面下率を示し、未測定点は線で埋めない。

学習処理の計測時間は7686.970秒。half予約で他runと資源を共有したため、所要時間から方式の計算効率の優劣は判断しない。学習開始時のcommit・patch・command・設定と原本照合用checksumを保存し、巨大checkpointはローカル出力に保持する。CLI移設前後の演算・解決済み設定の同値性は別の資格確認に記録している。再現コマンドは実行時の入口を維持し、後続変更に合わせて書き換えていない。

seed42の6条件が完了し、validation位置RMSEが最小の新方式である候補Bを追加seedの対象に選んだ。主対照とBのseed43/44を追加した計10 runの後に全長test・物理残差・欠損/イベント/view数の層別・実観測診断を行う。本登録時点ではtest推論も採用判断も行っていない。validationの近接した値だけではrollout項の効果やseed間の再現性を断定できない。

初回の同seed条件はCUDA unknown errorで中断したため、失敗記録run-i1044-reprojection-s42-cuda-interruptedと途中出力を保持し、同じseed・学習設定で最初から実行し直した。途中checkpointは引き継いでおらず、中断試行は成功runの合計10件に含めない。

本runの選定validation位置RMSEは、位置教師のみの条件より大きかった。これは固定4,000更新・seed42での観測であり、再投影損失が一般に不利とは結論しない。clean再投影・速度・rolloutなどの別の評価軸、全長testと追加seedを合わせて比較する。速度headは共通容量として存在するが速度教師を使っていないため、位置差分の運動指標と区別する。旧9,000更新checkpointはhead構成・予算の異なる参考値で、本runとの同一予算比較ではない。

CUDA診断を含むqueueログ原本はgzipで無損失保存し、展開後とqueue原本のSHA-256一致を検証した。queue-summary.logではdriver診断行だけを明示して省略する。
