---
id: run-blcs-residual-v2-balanced-s42
type: run
task: blcs
sequence: 37
recorded_at: '2026-09-21'
title: BLCS Court14残差v2・balanced loss（epoch23後native worker中断）
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: failed
config:
  recipe: train_triangulation_residual_v2
  loss_mode: balanced_regret
  feature_encoding: raw
  max_epochs: 30
  seed: 42
  train_scenes: 8000
  evaluation_views: 3
  workers: 6
  start_method: fork
  opencv_threads: 12
metrics:
  last_completed_epoch: 23
  completed_optimizer_steps: 6000
  last_val_world_mpjpe_m: 0.765331
  initial_val_world_mpjpe_m: 0.784012
  epoch24_sequential_probe_samples: 8000
  epoch24_sequential_probe_failures: 0
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.blcs.scripts.train_triangulation_residual --config-name
    train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=blcs/triangulation_residual_v2_balanced_s42_20260921
    run.seed=42 v2.loss_mode=balanced_regret training.trainer.enable_progress_bar=false
    && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.base.scripts.infer_triangulation_residual --task
    blcs --run-dir /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_balanced_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/blcs/balanced'
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-blcs-residual-v2-balanced-s42
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789944342617216859_322769_blcs-residual-v2-balanced-s42.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_balanced_s42_20260921/logs/version_0
  recovery_checkpoint: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v2_balanced_s42_20260921/resume_epoch23.ckpt
  recovery_checkpoint_sha256: f55c8c31e6713f6fa5de95519f8f1c161ab6baf4578796b386964923e3a1e3a2
  curves: knowledge/runs/run-blcs-residual-v2-balanced-s42/curves.png
  tb_logdir: outputs/blcs/triangulation_residual_v2_balanced_s42_20260921/logs/version_0
parents:
- run-blcs-residual-v2-gpu-smoke-r2
relations: []
papers: []
tags:
- triangulation-residual-v2
- balanced-regret
- native-worker-failure
---

## 考察 / Findings

### 要約
validationの平均3D誤差は初期値0.784012 mからepoch23で0.765331 mへ下がったが、その後のDataLoader workerがnative allocatorエラーで中断した。testと実clip推論は未実行のため、このrun単体を完走・精度検証済みとは扱わない。

### 失敗と切り分け
ログは`munmap_chunk(): invalid pointer`、worker PID324858のSIGABRT。RAM/shared memory不足の記録はない。同じ保存config、seed/index/epoch24をfresh単一processで全8000件走査し、Python/OpenCV例外もnative abortも再現しなかった。特定sample依存は支持されず、fork・OpenCV/IPP/pthreads・長寿命workerの累積状態が候補だが原因は未確定。CPU走査は並行性・実行順・24epochの蓄積を再現しない。

### 復旧
epoch23/global_step6000のlast checkpointを別名で保存し、worker0・OpenCV1threadで別queue runとして再開する。元logger directoryを保持してModelCheckpointの最良値・EarlyStopping・optimizer/schedulerを復元する。再開後のbatch順/RNG進行が無中断実行と完全一致するとは保証しない。サンプルのcorruptionはindex/epoch seedで固定され、GT誤差による選別は加えていない。

共有output_dirには後続resumeの成果物も書かれるため、このrunの証拠は固有log・再現bundle・退避checkpointとする。次の入力conditioning実験ではspawn workerとOpenCV1threadを使い、元クラッシュ解消の長時間保証とは区別する。
