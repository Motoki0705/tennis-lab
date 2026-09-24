---
id: run-plcs-residual-v2-asinh-legacy-s42
type: run
task: plcs
sequence: 112
recorded_at: '2026-09-21'
title: PLCS 再投影差分asinh conditioning・legacy loss対照
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  recipe: train_triangulation_residual_v2
  loss: legacy
  residual_encoding: asinh
  residual_scale: 0.01
  seed: 42
  max_epochs: 30
  selected_epoch: 29
  evaluation_views: 3
  worker_start: spawn
  opencv_worker_threads: 1
metrics:
  world_mpjpe_m: 0.12806656956672668
  initial_world_mpjpe_m: 0.1555238515138626
  root_error_m: 0.07925465703010559
  initial_root_error_m: 0.11979939043521881
  relative_mpjpe_m: 0.0985979214310646
  initial_relative_mpjpe_m: 0.15624013543128967
  test_world_median_m: 0.045033416210962046
  initial_world_median_m: 0.04613987371141297
  test_world_p95_m: 0.529473389193742
  test_sample_improved_fraction: 0.5440881763527055
  test_point_improved_fraction: 0.3732925409642815
  legacy_minus_asinh_mean_m: -0.008137151737701045
  legacy_minus_asinh_bootstrap95_m:
  - -0.009231631350829243
  - -0.006912681649741061
  real_after_reprojection_px: 3.650175970898959
repro:
  commit: 771649c98814914f4a9164cf4ca6df44aabc9041
  branch: codex/residual-v2-conditioning
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual
    --config-name train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=plcs/triangulation_residual_v2_asinh_legacy_s42_20260921
    run.seed=42 v2.loss_mode=legacy features.residual_encoding=asinh features.residual_scale=0.01
    training.trainer.enable_progress_bar=false && CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m
    src.tasks.base.scripts.infer_triangulation_residual --task plcs --run-dir /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_asinh_legacy_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/plcs/asinh'
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-plcs-residual-v2-asinh-legacy-s42
  predictions: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_asinh_legacy_s42_20260921/predictions/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789950510471956561_1101467_plcs-residual-v2-asinh-legacy-s42.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_asinh_legacy_s42_20260921/logs/version_0
  real_comparison: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/plcs/asinh/index.html
  curves: knowledge/runs/run-plcs-residual-v2-asinh-legacy-s42/curves.png
  tb_logdir: outputs/plcs/triangulation_residual_v2_asinh_legacy_s42_20260921/logs/version_0
parents:
- run-plcs-residual-v2-legacy-s42
- run-residual-v2-conditioning-cpu
relations: []
papers: []
tags:
- triangulation-residual-v2
- input-conditioning
- asinh
- negative-result
---

## 考察 / Findings

### 要約
同じlegacy lossでresidual featureだけasinh(r/0.01)へ変更。val0.119650、test平均0.128067mで、raw対照のval0.111298/test0.119929mより悪かった。CPUでのBF16感度改善は、PLCSの精度改善には直結しなかった。既定encodingはrawを維持する。

### 比較と限界
全30epoch、同じseed42/GT/初期3D/mask/familyを使用し、保存test配列の一致を確認。raw−asinhのtest差は−0.008137m、scene bootstrap95%範囲[−0.009232,−0.006913]m。これはこの学習seed内のscene変動のみで、再学習seedの不確実性ではない。runtimeはspawn/OpenCV1threadへ変更したが、同じseedのsampleとepoch共有の一致を別途確認した。

### 実clip
平均再投影は3.252870→3.650176px。1m超の四肢は補完込み49→2件だが、長い前腕は残る。独立3D正解がないため、raw対照より小さい再投影だけでこのモデルを選ばない。採否はvalidationで決め、test/実clipでscaleを調整していない。
