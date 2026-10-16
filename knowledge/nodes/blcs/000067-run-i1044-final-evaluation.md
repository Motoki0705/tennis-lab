---
id: run-i1044-final-evaluation
type: run
task: blcs
sequence: 67
recorded_at: '2026-10-14'
title: '#1044固定10runの全長test・実観測評価と保存再計算'
issue: 1044
provider: codex
date: '2026-10-14'
status: done
config:
  protocol: issue1044-v1
  split: test
  training_runs: 10
  optimizer_updates_per_run: 4000
  used_test_for_selection: false
  synthetic_conditions: 16
  real_conditions: 12
  test_scenes_per_condition: 100
  test_frames_per_condition: 35704
  window_frames: 128
  overlapping_windows: false
  selected_arm: parameters
  selection_seed: 42
  confirmation_seeds:
  - 43
  - 44
  independent_real_3d_gt: false
  dataset_manifest_verified_files: 59006
metrics:
  control_test_trajectory_rmse_seed_mean_m: 1.1458906379421163
  candidate_test_trajectory_rmse_seed_mean_m: 0.9302763101992948
  saved_synthetic_recomputations: 16
  saved_real_recomputations: 12
artifacts:
  run_dir: knowledge/runs/run-i1044-final-evaluation
  output_dir: outputs/blcs/i1044
  log: knowledge/runs/run-i1044-final-evaluation/queue-summary.log
  raw_log: knowledge/runs/run-i1044-final-evaluation/queue.log.gz
  log_archive: knowledge/runs/run-i1044-final-evaluation/log-archive.json
  capture_checksums: knowledge/runs/run-i1044-final-evaluation/capture_checksums.json
  artifact_manifest: knowledge/runs/run-i1044-final-evaluation/artifact_manifest.json
  completion: knowledge/runs/run-i1044-final-evaluation/evaluation-complete.json
  training_receipts: knowledge/runs/run-i1044-final-evaluation/completed-training.json
  selection: knowledge/runs/run-i1044-final-evaluation/additional-seed-selection.json
  dataset_verification: knowledge/runs/run-i1044-final-evaluation/final-dataset-verification.json
  synthetic_predictions: knowledge/runs/run-i1044-final-evaluation/evaluations
  real_predictions: knowledge/runs/run-i1044-final-evaluation/real
  report: knowledge/runs/run-i1044-final-evaluation/report
  curves: knowledge/runs/run-i1044-final-evaluation/report/validation_learning.png
  saved_artifacts_audit: knowledge/runs/run-i1044-final-evaluation/saved-artifacts-audit.json
  report_regeneration_audit: knowledge/runs/run-i1044-final-evaluation/report-regeneration-audit.json
parents:
- group-i1044-seed42-selection
relations:
- to: run-i1044-position-s42-u4000
  rel: evaluates
- to: run-i1044-reprojection-s42-u4000
  rel: evaluates
- to: run-i1044-velocity-s42-u4000
  rel: evaluates
- to: run-i1044-rollout-position-s42-u4000
  rel: evaluates
- to: run-i1044-rollout-velocity-s42-u4000
  rel: evaluates
- to: run-i1044-parameters-s42-u4000
  rel: evaluates
- to: run-i1044-reprojection-s43-u4000
  rel: evaluates
- to: run-i1044-parameters-s43-u4000
  rel: evaluates
- to: run-i1044-reprojection-s44-u4000
  rel: evaluates
- to: run-i1044-parameters-s44-u4000
  rel: evaluates
- to: run-i1032-blcs-physics-v3-e200-s42
  rel: evaluates
papers: []
tags:
- issue1044
- full-test
- saved-recompute
- real-observation-diagnostics
session: 01a1327c-d543-7673-8143-2764d9f62447
repro:
  commit: 9948985912f3f2dd1a62f212b30fd92db8e57f2e
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv/bin/python -m src.tasks.blcs.experiments.scripts.final_evaluation --data-root
    /home/kamimura/projects/tennis-lab/data --historical-checkpoint /home/kamimura/projects/tennis-lab/ckpt/blcs/axial-base-physics-v3-kp14-t128-v3-6-e200-s42-epoch189.ckpt
    --real-store /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-default-meiji-clip000-20260927/store
---



全10 runの学習とvalidationによるcheckpoint・追加seed対象の固定後に、共有queueのall予約で最終評価を実行した。queueはdone、完了receiptも同じrun IDを指す。各学習runの4,000更新・seed/arm・checkpoint SHA-256・test未使用を確認し、固定snapshot全59,006ファイルの内容と集合が既存manifestに一致してからtest推論へ進んだ。

合成testは全100 scene・35,704 frameを16条件で評価した。10件の学習run、旧9,000更新重み、GT・幾何補間・jitter・time shift・collision crossingが含まれる。入力は固定128frameの非重複窓で全長を覆い、連結した出力に同じ指標を適用した。短い最終窓も含める。実観測は固定Meiji clipを12条件（11 checkpointと幾何補間）で診断した。

各synthetic条件にevaluation.json、pred_test.npz、recomputed.jsonを、各real条件にも観測・投影行列・予測・再計算結果を保存した。元のcheckpointやデータstoreを使わない再計算を実行し、保存JSONとのキー・支持数・nullと数値（rtol=atol=1e-6）の一致を検査した。syntheticでは全体/clip別・層別、BのGT区間/予測区間積分とevent・ODE fit診断を含む。値が未定義の指標をゼロで補わない。これは保存された積分位置・速度から指標を再計算する検査で、積分器自体の再実行による配列一致検査ではない。保存済みoriginal/recomputed JSONの28条件と層別368行の照合結果、queue終了状態の根拠はsaved-artifacts-audit.jsonに記録した。

実行開始時のcommit・command・patchを原本のまま保存した。入力receipt、最終receipt、全評価のJSON/NPZと16件のreport（図7枚を含む）をコピーし、各原本とのSHA-256一致をartifact_manifest.jsonに記録した。巨大checkpointはGitへ複製していない。CUDA driver診断を含むqueueログは無損失gzipと展開後SHAで保持し、可読要約で省略した行の条件・件数をlog-archive.jsonに明示した。

このrunは追加学習をしない。学習曲線と出力勾配の図は10件の学習TensorBoardから生成したもので、元の全scalarと個別runの再現記録は各学習ノードにある。図表再生成用に20個の元TensorBoard eventファイルも、このbundle内で元と同じrun相対パスに保存した。元の学習出力・checkpoint・data/storeを入力にせず、一時出力先でreportを再生成し、16ファイルすべてのSHA-256が元reportと一致した（report-regeneration-audit.json）。旧9,000更新重みは予算・head構成が異なる参考対照で、同条件の主対照はreprojectionである。実観測は保存注釈と校正由来のcourt点を用い、独立3D GTがないため実3D精度の改善を結論しない。

主対照とBの3seed平均test軌道RMSEは1.145891 mと0.930276 mだった。seed42は方式選択にも用いたため、seed43/44の確認結果と分けて解釈する。段階比較、支持数、退行、積分出力の限界と次の実験案は[最終比較group](000068-group-i1044-progressive-losses-u4000.md)に集約した。採用閾値は事前に設けず、本番checkpointも置換しない。
