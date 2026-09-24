---
id: run-plcs-residual-v2-legacy-s42
type: run
task: plcs
sequence: 110
recorded_at: '2026-09-21'
title: PLCS Court14校正・持続誤検出v2の従来損失対照
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
  world_mpjpe_m: 0.11992941796779633
  initial_world_mpjpe_m: 0.1555238515138626
  root_error_m: 0.07734089344739914
  initial_root_error_m: 0.11979939043521881
  relative_mpjpe_m: 0.08812686055898666
  initial_relative_mpjpe_m: 0.15624013543128967
  test_world_median_m: 0.04412063843884134
  initial_world_median_m: 0.04613987371141297
  test_world_p95_m: 0.49165841094811136
  initial_world_p95_m: 0.6624315330662105
  test_sample_improved_fraction: 0.5501002004008017
  test_point_improved_fraction: 0.39539649151243667
  test_central99_mean_gain_m: 0.022812410271451443
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual --config-name
    train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=plcs/triangulation_residual_v2_legacy_s42_20260921
    run.seed=42 v2.loss_mode=legacy training.trainer.enable_progress_bar=false &&
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.base.scripts.infer_triangulation_residual --task
    plcs --run-dir /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_legacy_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/plcs/legacy'
    --device cuda --no-render
artifacts:
  run_dir: knowledge/runs/run-plcs-residual-v2-legacy-s42
  predictions: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_legacy_s42_20260921/predictions/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789944342809906724_322820_plcs-residual-v2-legacy-s42.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_legacy_s42_20260921/logs/version_0
  real_comparison: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/plcs/legacy/metrics.json
  curves: knowledge/runs/run-plcs-residual-v2-legacy-s42/curves.png
  tb_logdir: outputs/plcs/triangulation_residual_v2_legacy_s42_20260921/logs/version_0
parents:
- run-plcs-residual-v2-gpu-smoke-r2
relations:
- to: run-plcs-residual-v2-balanced-s42
  rel: compares
papers: []
tags:
- triangulation-residual-v2
- legacy-loss
- paired-loss-comparison
---

## 考察 / Findings

### 要約
新しい合成testで平均3D誤差0.155524→0.119929 m、中央値0.046140→0.044121 m、p95 0.662432→0.491658 m。validation最良epoch29を採用した。同じv2データのbalanced loss（val0.122009、test0.131759 m）より、この従来loss対照が良かった（val0.111298 m）。

### 条件と解釈
Court14再推定、四隅+正面2候補、持続誤検出は共通で、損失だけ従来の成分別Smooth-L1へ戻した。新損失bundleの優位は支持されず、個々のregret/stratum/vector Huberの寄与は未分離。改善sample率55.0%、point率39.5%。最大補正1%を除いた平均gainも+0.022812 mで、裾だけの改善ではないが、小さな誤差を持つ多数点の悪化は残る。

### 実clipと次の比較
Meijiの再投影平均3.252870→3.739684 px、補完を含む1m超四肢は49→2件、最大1.116m。独立3D正解はなく、異常低減が真の位置精度を保証しない。同じlegacy lossを固定し、次はasinh入力conditioningだけを比較する。v1とは誤差生成・view条件が異なるため直接の性能比較はしない。
