---
id: run-i991-i1014-coordinates-report-20261005-v2
type: run
task: ball_refiner
sequence: 39
recorded_at: '2026-10-05'
title: 座標Refiner 15条件の入力同一性・保存予測再集計・単独GPU速度
issue:
- 991
- 1014
provider: codex
session: 01a10697-63da-75b2-ae84-f2980da51c23
date: '2026-10-05'
status: done
config:
  training_runs: 15
  device: NVIDIA GeForce RTX 5060 Ti
  latency_protocol: exclusive queue; batch=1,T=128; 5 warmups,20 timed calls; sync
    CUDA; exclude model load and transfers
  manifest_sha256: 18fe4f79dcc1a030edc41922e61b15085a0de0bd25735c50a4a5e0ec8dd87cca
metrics:
  selected_2d_test_rmse_px: 15.188450813293457
  selected_3d_test_rmse_m: 0.4249493479728699
repro:
  commit: 0bdeb9514ae1e780f1d8e8e08a431b2aa9f5143c
  branch: codex/coordinate-ball-refiners-991-1014
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python
    tests/benchmarks/ball_refiner_coordinates_report.py --runs-root /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train
    --output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/coordinates/20261005-v2
    --tag 20261005-v2 --device cuda
artifacts:
  run_dir: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791166823083491190_2779389_i991-i1014-coordinates-report-20261005-v2.log
  comparison: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2/comparison.json
  dataset_audit: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2/dataset_audit.json
  physical_diagnostics: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2/physical_diagnostics.json
  cli_examples: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2/cli_examples
parents:
- run-i991-coords-2d-gan-p025-s42-20261005-v2
- run-i991-coords-2d-gan-p05-s42-20261005-v2
- run-i991-coords-2d-gan-p075-s42-20261005-v2
- run-i991-coords-2d-regression-p025-s42-20261005-v2
- run-i991-coords-2d-regression-p05-s42-20261005-v2
- run-i991-coords-2d-regression-p075-s42-20261005-v2
- run-i1014-coords-3d-flow-p025-s42-20261005-v2
- run-i1014-coords-3d-flow-p05-s42-20261005-v2
- run-i1014-coords-3d-flow-p075-s42-20261005-v2
- run-i1014-coords-3d-gan-p025-s42-20261005-v2
- run-i1014-coords-3d-gan-p05-s42-20261005-v2
- run-i1014-coords-3d-gan-p075-s42-20261005-v2
- run-i1014-coords-3d-regression-p025-s42-20261005-v2
- run-i1014-coords-3d-regression-p05-s42-20261005-v2
- run-i1014-coords-3d-regression-p075-s42-20261005-v2
relations: []
papers: []
tags:
- coordinate-refiner
- ablation
- evaluation
- isolated-latency
---

15本の完了済み学習runについて、dataset manifest・rally split・更新予算・評価seedと劣化条件が一致することを確認した。同一次元の評価入力hashが全方式で同一で、保存NPZから再集計した全体・欠損・観測・イベント近傍のmetricは記録値と一致した。このrunは新しい学習やtestの再選定を行わず、各runでvalidation選択したcheckpointと保存予測を比較する。

共有queueのall枠でGPUを確保し、各best checkpointのmodel.predictをbatch=1・128frame、warmup 5回・計測20回で測定した。各計測前後にCUDA同期し、checkpoint load・host/device転送・三角測量は時間に含めない。中央値・P95はcomparison.jsonに記録した。

kg_curvesはこの実装のtrain/reconstruction・val/rmseのtagに対応せずskipしたため、保存JSONLとvalidation JSONから別軸で曲線を作成した。各学習runに曲線と元の数値を添付した。CLI例は学習済み2D回帰と、同一3D入力による回帰・Flowで、観測区間も含む全462frameの有限な出力を確認した。2DのCPU CLIと同じCPU APIはbit一致し、GPUの保存予測との最大差は0.031pxだった。CPU/CUDA間のbit一致は要求していない。

dataset交換receiptと生成audit、現在／再投影前のmanifest・設定も保持した。physical_diagnostics.jsonは保存済み3D予測の負高さの追加診断で、checkpoint選択やモデルの再調整には使用していない。統合した判断は比較group nodeを参照する。
