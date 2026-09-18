---
id: run-slcs-plcs-meiji-foot-e60-resume-v2
type: run
title: Meiji PLCS 60epoch学習・区間2の環境中断
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: multiview_axial_foot_residual
  loss: all_outputs_beta01_reprojection
  data: camera_view_real_rgb_ft_v1
metrics:
  saved_epoch_index: 46
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_meiji_foot_real_rgb
    run.output_dir=plcs/train/meiji_foot_real_rgb/s42-001 run.init_weights=null run.resume=plcs/train/meiji_foot_real_rgb/s42-001/logs/version_1/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v2
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs/version_2
  tb_logdir: outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs/version_2
  curves: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v2/curves.png
parents:
- run-slcs-plcs-meiji-foot-e60-resume-v1
relations: []
tags:
- slcs
- plcs
- meiji
- interrupted
---

## 考察 / Findings

### 要約
WSL再起動によりepoch index 46で中断。60epochの完了runとして扱わない。

### アーキテクチャ詳細
foot-residual PLCSを既存source-motion分離合成データで適応。構成・再現コマンドはrepro bundleに保存。

### メトリクスの解釈
saved_epoch_indexは0始まりの保存済み学習状態。test評価は未実施。指標は合成教師との一致であり、実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察
環境中断からモデルの優劣は結論できない。

### 既存実験との比較
同一60epoch実験の再開区間。各区間のvalidation最良を横断して最終選定する。

### 次に有効な実験
last checkpointのoptimizer/scheduler状態を保持し、総60epochまで継続する。
