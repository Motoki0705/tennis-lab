---
id: run-blcs-residual-v2-balanced-s42-resume
type: run
task: blcs
sequence: 40
recorded_at: '2026-09-21'
title: BLCS Court14残差v2 balanced loss・checkpoint復旧と最終評価
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  recipe: train_triangulation_residual_v2
  loss: balanced_regret
  features: raw
  seed: 42
  resume_epoch: 23
  resume_global_step: 6000
  final_epoch: 29
  selected_epoch: 29
  workers: 0
  opencv_threads: 1
  logger_history: same logs/version_0
  evaluation_views: 3
metrics:
  world_mpjpe_m: 0.8124020099639893
  initial_world_mpjpe_m: 0.844004213809967
  root_error_m: 0.8124020099639893
  initial_root_error_m: 0.844004213809967
  relative_mpjpe_m: 0.0
  initial_relative_mpjpe_m: 0.0
  test_world_median_m: 0.08059336764033603
  initial_world_median_m: 0.07926252226844896
  test_world_p95_m: 4.6545722864282135
  initial_world_p95_m: 4.935474182305183
  test_sample_improved_fraction: 0.564
  test_point_improved_fraction: 0.40104549794187155
  test_central99_mean_gain_m: 0.02390982808369902
  real_before_reprojection_px: 2.301307211973855
  real_after_reprojection_px: 2.305066556764045
  real_mean_correction_m: 0.0019174086628481746
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python knowledge/runs/run-blcs-residual-v2-balanced-s42-resume/resume.py
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=blcs/triangulation_residual_v2_balanced_s42_20260921
    run.resume=blcs/triangulation_residual_v2_balanced_s42_20260921/resume_epoch23.ckpt
    run.seed=42 v2.loss_mode=balanced_regret data.num_workers=0 training.trainer.enable_progress_bar=false
    && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.base.scripts.infer_triangulation_residual
    --task blcs --run-dir /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_balanced_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/balanced'
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-blcs-residual-v2-balanced-s42-resume
  predictions: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_balanced_s42_20260921/predictions/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789947410149998906_730599_blcs-residual-v2-balanced-s42-resume.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_balanced_s42_20260921/logs/version_0
  real_comparison: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/balanced/index.html
  curves: knowledge/runs/run-blcs-residual-v2-balanced-s42-resume/curves.png
  tb_logdir: outputs/blcs/triangulation_residual_v2_balanced_s42_20260921/logs/version_0
parents:
- run-blcs-residual-v2-balanced-s42
relations:
- to: run-blcs-residual-v2-legacy-s42
  rel: compares
papers: []
tags:
- triangulation-residual-v2
- balanced-regret
- resumed
- paired-loss-comparison
---

## 考察 / Findings

### 要約
保存したepoch23/global_step6000からworker0/OpenCV1threadで再開し、epoch29まで完走。val最良0.753288mのcheckpointでtest平均0.844004→0.812402m（約3.7%改善）、p95 4.935474→4.654572m。中央値は0.079263→0.080593mと微悪化した。

### 比較の意味
同じtest scene、GT、初期3D、frame/init mask、severity/familyがlegacy runとバイト一致することを確認。legacyの平均0.826194mより良い一方、point中央値はlegacyよりも僅かに悪い。改善scene率56.4%、point率40.1%。最大補正1%の総gain寄与は25.1%、残り99%も平均+0.023910mであり、改善は裾1%だけではない。それでも多数pointの小さな悪化は残る。

### 復旧条件と限界
元logger/checkpoint directoryを固定し、ModelCheckpointのbest history、EarlyStopping、optimizer/schedulerを引き継いだ。loss/data/modelの定義は変更していない。無中断実行と後半のbatch順・RNG進行が完全一致する保証はなく、単一seedのloss効果の断定は避ける。曲線は同じTensorBoard historyにある中断前と復旧後を含む。元のnative故障は親nodeを参照。

### 実clipと次の検証
Meijiの再投影平均2.301307→2.305067px、補正平均1.92mmで、実用的な位置改善は確認できない。独立3D正解はない。固定asinh入力conditioningをlegacy loss対照で別に比較し、中央値・初期誤差bin・event中/外と実clipを評価する。production checkpointは更新しない。
