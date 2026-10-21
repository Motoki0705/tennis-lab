---
id: group-i1044-seed42-selection
type: group
task: blcs
sequence: 61
recorded_at: '2026-10-13'
title: 6条件のvalidation選択完了・追加seedは物理head条件B
issue: 1044
members:
- run-i1044-position-s42-u4000
- run-i1044-reprojection-s42-u4000
- run-i1044-velocity-s42-u4000
- run-i1044-rollout-position-s42-u4000
- run-i1044-rollout-velocity-s42-u4000
- run-i1044-parameters-s42-u4000
parents:
- group-i1044-qualification
papers: []
tags:
- issue1044
- progressive-loss-ablation
- validation-only
config:
  protocol: issue1044-v1
  seed42_optimizer_updates_per_run: 4000
  checkpoint_selection: minimum validation trajectory RMSE
  additional_seed_selection: minimum checkpoint validation trajectory_rmse_m
  eligible_new_arms:
  - velocity
  - rollout_position
  - rollout_velocity
  - parameters
  selected_arm: parameters
  confirmation_seeds:
  - 43
  - 44
  confirmation_arms:
  - reprojection
  - parameters
  used_test: false
  adoption_decision: false
  stage_complete: seed42 training and additional-seed selection
  full_10_run_comparison_complete: false
metrics:
  selected_validation_trajectory_rmse_m:
    position: 0.6985505819320679
    reprojection: 0.8476958870887756
    velocity: 0.7142558097839355
    rollout_position: 0.7140101194381714
    rollout_velocity: 0.6923530697822571
    parameters: 0.6518378257751465
repro:
  commit: a1921e349b1ec6162affccc6b96f38c4af2ad729
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv/bin/python -m src.tasks.blcs.experiments.scripts.repeat_selected --select-only
session: 01a1327c-d543-7673-8143-2764d9f62447
artifacts:
  run_dir: knowledge/runs/group-i1044-seed42-selection
  log: knowledge/runs/group-i1044-seed42-selection/queue.log
  selection: knowledge/runs/group-i1044-seed42-selection/additional-seed-selection.json
  capture_checksums: knowledge/runs/group-i1044-seed42-selection/capture_checksums.json
  members: knowledge/runs/group-i1044-seed42-selection/members.json
  selected_validation_values: knowledge/runs/group-i1044-seed42-selection/selected-validation-values.json
  selected_validation_table: knowledge/runs/group-i1044-seed42-selection/selected-validation-values.csv
  selection_plot: knowledge/runs/group-i1044-seed42-selection/selected-validation-rmse.png
  selection_survey: knowledge/runs/group-i1044-seed42-selection/selection-survey.json
---

seed42の6条件が各4,000 optimizer updatesを完了し、事前規定のvalidation位置RMSE最小で追加seedの対象を候補B（parameters）に決めた。新方式の候補集合はvelocity、rollout_position、rollout_velocity、parametersの4条件で、位置のみ・既存戦略の主対照はこの候補集合に含めない。4条件の順位はB、A、rollout位置、速度教師だった。採用判断ではなく、主対照reprojectionとBのseed43/44による確認実験を進める決定である。合計10 runのうち、このノードが記録する完了範囲は最初の6 runと追加seed対象の選択である。

以下は全100 validation sceneの中心128frame（短いsceneは全長）でcheckpointを選んだときの値で、全長testではない。全runの更新予算は4,000であり、最小値を得たcheckpoint時点は別に記す。

| 条件（前段への追加） | 選定validation軌道RMSE [m] | 選定checkpoint更新数 |
|---|---:|---:|
| 位置教師のみ | 0.698551 | 4000 |
| ＋clean再投影（既存戦略の主対照） | 0.847696 | 4000 |
| ＋速度教師 | 0.714256 | 4000 |
| ＋rollout位置整合 | 0.714010 | 4000 |
| ＋rollout速度整合（A） | 0.692353 | 4000 |
| ＋最小物理headと補助損失群（B） | 0.651838 | 3825 |

位置のみからclean再投影を加えた主対照では、このseed・短予算で位置RMSEが大きくなった。そこから速度教師、rollout位置、rollout速度、Bのheadと補助損失群を加えたときの点推定は順に小さくなった。rollout位置を加えた差は約0.000246 mと小さく、この1 seedの値だけで効果を断定しない。Bはheadと複数損失を一群で追加するため、各補助項の独立効果は分離できない。位置RMSEだけから再投影・運動・物理整合すべての優劣は決められない。

同じ選定時点の速度head、位置差分の速度/加速度、rollout、clean再投影、地面下率などの学習時validation scalarをCSV/JSONに保存した。全条件で同じ速度headを置くが、位置のみ・主対照の速度headには教師を与えないため、位置差分の運動指標と区別する。これらのTensorBoard scalarには指標別の支持数は保存されていない。母数と層別結果を含む評価は、全10 run終了後に全長testと実観測診断の保存NPZ/JSONで行う。値はTensorBoardに保存された精度のままであり、最終評価のfloat64再集計とは別の記録である。

選定validationの運動診断では、主対照の速度head RMSEは14.016 m/s、Bは2.572 m/sだった一方、rollout速度残差は主対照0.870 m/s、Bは1.106 m/sだった。速度教師なしでも整合残差が小さくなる例なので、残差単独を正確な運動の証拠にはしない。位置差分の加速度RMSEはAの75.268 m/s²に対してBは81.207 m/s²で、位置精度と全運動指標が一様に改善するわけではない。clean再投影RMSEは主対照125.816 px、位置のみ23.798 px、Bは20.473 pxだった。集約値だけでは外れ値や無効射影の寄与を特定できず、最終保存評価で支持数・clip別結果・保存予測を確認する。これらも選定に使ったvalidationの点推定であり、独立testの効果量ではない。

追加seed選択はall予約で最初の6 runが終わるのを待ち、6件の選定台帳を保存した上で実行された。選定台帳、各runのcheckpoint hash、元のqueue command・commit・patchとchecksumを保持した。testは候補選択に使っていない。seed42はcheckpointと方式の両方の選択に使うため、seed43/44の確認結果と区別して解釈する。旧9,000更新checkpointは予算とhead構成が異なる参考対照であり、実観測には独立3D GTがない。採用閾値は設けず、全実験結果を観測してから人間が判断する。
