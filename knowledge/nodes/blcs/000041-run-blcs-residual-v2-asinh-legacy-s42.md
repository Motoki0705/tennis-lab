---
id: run-blcs-residual-v2-asinh-legacy-s42
type: run
task: blcs
sequence: 41
recorded_at: '2026-09-21'
title: BLCS 再投影差分asinh conditioning・legacy loss対照
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
  world_mpjpe_m: 0.8126410245895386
  initial_world_mpjpe_m: 0.844004213809967
  root_error_m: 0.8126410245895386
  initial_root_error_m: 0.844004213809967
  relative_mpjpe_m: 0.0
  initial_relative_mpjpe_m: 0.0
  test_world_median_m: 0.0791901594966585
  initial_world_median_m: 0.07926252226844896
  test_world_p95_m: 4.784064360189415
  test_sample_improved_fraction: 0.565
  test_point_improved_fraction: 0.41708794390621823
  legacy_minus_asinh_mean_m: 0.01355329715977787
  legacy_minus_asinh_bootstrap95_m:
  - 0.004006234210904276
  - 0.023762304446516662
  real_after_reprojection_px: 2.4592646736281987
repro:
  commit: 771649c98814914f4a9164cf4ca6df44aabc9041
  branch: codex/residual-v2-conditioning
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.blcs.scripts.train_triangulation_residual
    --config-name train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=blcs/triangulation_residual_v2_asinh_legacy_s42_20260921
    run.seed=42 v2.loss_mode=legacy features.residual_encoding=asinh features.residual_scale=0.01
    training.trainer.enable_progress_bar=false && CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m
    src.tasks.base.scripts.infer_triangulation_residual --task blcs --run-dir /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_asinh_legacy_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/asinh'
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-blcs-residual-v2-asinh-legacy-s42
  predictions: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_asinh_legacy_s42_20260921/predictions/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789950510646529794_1101566_blcs-residual-v2-asinh-legacy-s42.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_asinh_legacy_s42_20260921/logs/version_0
  real_comparison: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/asinh/index.html
  curves: knowledge/runs/run-blcs-residual-v2-asinh-legacy-s42/curves.png
  tb_logdir: outputs/blcs/triangulation_residual_v2_asinh_legacy_s42_20260921/logs/version_0
parents:
- run-blcs-residual-v2-legacy-s42
- run-residual-v2-conditioning-cpu
relations: []
papers: []
tags:
- triangulation-residual-v2
- input-conditioning
- asinh
- mixed-result
---

## 考察 / Findings

### 要約
asinh(r/0.01)によりlegacy lossのtest平均はraw対照0.826194→0.812641mへ改善。中央値0.079190mは初期値0.079263mとほぼ同じで、改善は小さい。val0.756531mはbalanced/rawの0.753288mに届かず、既定はbalanced/rawとする。

### 比較と限界
同じseed/GT/初期3D/mask/familyを保存test配列で照合した。raw−asinh平均差+0.013553m、scene bootstrap95%範囲[+0.004006,+0.023762]m。入力は同一、feature変換だけを明示的に変えた。runtimeはspawn/OpenCV1threadで、sample一致とepoch共有を確認。30epochをnative worker crashなく完走したが、元の故障原因を断定する実験ではない。

### 実clipと判断
平均再投影は2.301307→2.459265pxへ増え、balanced/rawの2.305067pxより大きい。実clipの3D正解はなく、実用的改善は未確立。入力差分の埋め込み感度を高める効果と、精度・汎化の効果を分けて扱う。新損失＋asinhの交互作用や複数seedは未検証。
