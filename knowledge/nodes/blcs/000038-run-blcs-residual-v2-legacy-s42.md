---
id: run-blcs-residual-v2-legacy-s42
type: run
task: blcs
sequence: 38
recorded_at: '2026-09-21'
title: BLCS Court14校正・持続誤検出v2の従来損失対照
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  recipe: train_triangulation_residual_v2
  loss: legacy
  features: raw
  seed: 42
  max_epochs: 30
  selected_epoch: 29
  train_views: 2..6
  evaluation_views: 3
  camera_error: Court14 -> f/PnP fit
  persistent_errors: true
metrics:
  world_mpjpe_m: 0.8261942863464355
  initial_world_mpjpe_m: 0.844004213809967
  root_error_m: 0.8261942863464355
  initial_root_error_m: 0.844004213809967
  relative_mpjpe_m: 0.0
  initial_relative_mpjpe_m: 0.0
  test_world_median_m: 0.0798757857683447
  initial_world_median_m: 0.07926252226844896
  test_world_p95_m: 4.788362130816853
  initial_world_p95_m: 4.935474182305183
  test_sample_improved_fraction: 0.421
  test_point_improved_fraction: 0.3632322789663969
  test_central99_mean_gain_m: 0.008446521036475519
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.blcs.scripts.train_triangulation_residual --config-name
    train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=blcs/triangulation_residual_v2_legacy_s42_20260921
    run.seed=42 v2.loss_mode=legacy training.trainer.enable_progress_bar=false &&
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.base.scripts.infer_triangulation_residual --task
    blcs --run-dir /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_legacy_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/legacy'
    --device cuda --no-render
artifacts:
  run_dir: knowledge/runs/run-blcs-residual-v2-legacy-s42
  predictions: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_legacy_s42_20260921/predictions/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789944342966889046_322860_blcs-residual-v2-legacy-s42.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_legacy_s42_20260921/logs/version_0
  real_comparison: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/legacy/metrics.json
  curves: knowledge/runs/run-blcs-residual-v2-legacy-s42/curves.png
  tb_logdir: outputs/blcs/triangulation_residual_v2_legacy_s42_20260921/logs/version_0
parents:
- run-blcs-residual-v2-gpu-smoke-r2
relations:
- to: run-blcs-residual-v2-balanced-s42
  rel: compares
papers: []
tags:
- triangulation-residual-v2
- legacy-loss
- paired-loss-comparison
---

## 考察 / Findings

### 要約
新しい合成testで平均3D誤差0.844004→0.826194 m（約2.1%改善）、p95 4.935474→4.788362 m。一方、中央値は0.079263→0.079876 mと微悪化した。validation最良epoch29（0.767982 m）を選択。大半への有効な補正はまだ確認できない。

### 条件と解釈
新v2データと同じモデルを使い、旧来のroot/world重複を含む成分別Smooth-L1を対照として保持した。改善sample率42.1%、point率36.3%。補正中央値は3.94mmで、最大補正1%が総gainの53.0%を占める。残り99%にも+0.008447mの平均gainはあるが、全体平均だけで学習停滞解消と判断しない。

### 実clipと次の比較
Meijiの平均再投影は2.301307→2.320818 px、平均補正4.58mm。独立3D正解がなく、実用的な改善の根拠は弱い。balanced runはnative worker中断後の復旧を別nodeに記録する。同じlegacy lossのまま入力conditioningを比較し、GT値でscaleやcheckpointを選ばない。
