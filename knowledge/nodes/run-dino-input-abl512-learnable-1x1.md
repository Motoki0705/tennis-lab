---
id: run-dino-input-abl512-learnable-1x1
type: run
title: DINO入力 learnable 1x1（512 step）
provider: codex
session: 01a06b47-7972-7521-a83c-be95cccf7d91
date: '2026-09-04'
status: done
config:
  model: dino-swin-l-4scale-lora-r8-learnable-1x1
  loss: detr-focal-aabb-giou-scale-axial
  data: procedural-court-800-identity-512
metrics:
  instance_precision: 0.134021
  instance_recall: 0.134021
  instance_f1: 0.134021
  instance_count_accuracy: 1.0
  instance_count_mae: 0.0
  matched_center_mean_error_px: 0.445906
  matched_scale_relative_error: 0.099535
  matched_axial_angle_mean_error_deg: 5.13174
  matched_corner_mean_error_px: 22.093279
repro:
  commit: 6631f8224a1869bf1ef5d08c8bfbae41cc0cb3a4
  branch: feat/dino-detr-court-alignment
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/dino-detr-court-alignment
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.court_alignment.scripts.train
    --config-name train_dino paths.project_root=/home/kamimura/projects/tennis-lab
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party paths.output_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/dino-detr-court-alignment/outputs
    model.repository=/home/kamimura/projects/tennis-lab/third_party/DINO model.checkpoint_path=/home/kamimura/projects/tennis-lab/ckpt/dino/checkpoint0029_4scale_swin.pth
    model.device=cuda:0 data.train_samples=512 data.val_samples=128 data.test_samples=128
    data.batch_size=1 data.num_workers=2 training.steps_per_epoch=512 training.warmup_steps=32
    training.trainer.max_epochs=1 training.trainer.precision=32-true training.trainer.log_every_n_steps=25
    training.checkpoint.save_top_k=1 training.checkpoint.save_last=false run.fast_dev_run=false
    run.test_after_fit=true model.input_mode=learnable_1x1 run.output_dir=court_alignment/dino_input_ablation_512_learnable_1x1
artifacts:
  run_dir: knowledge/runs/run-dino-input-abl512-learnable-1x1
  predictions: knowledge/runs/run-dino-input-abl512-learnable-1x1/pred_test.npz
  output_dir: outputs/court_alignment/dino_input_ablation_512_learnable_1x1/logs/version_0
  curves: knowledge/runs/run-dino-input-abl512-learnable-1x1/curves.png
  tb_logdir: outputs/court_alignment/dino_input_ablation_512_learnable_1x1/logs/version_0
parents: [run-dino-input-abl512-repeat-rgb]
relations: []
tags: [court-alignment, dino, detr, lora, learnable-1x1, input-ablation]
---

## 考察 / Findings

### 要約

RGB複製で初期化したlearnable 1×1 adapterは、512 step時点でF1 `0.134021`、test loss `11.9208`となり、3条件中もっとも低い短期性能だった。

### アーキテクチャ詳細

DINO/LoRA/OBB head・損失・800×800データ条件は`repeat_rgb`と同一で、入力直前だけをtrainable `Conv2d(1,3,1)`へ変更した。weightは3channelすべて1、biasは0で、開始時点の写像は単純RGB複製と一致する。seed 42、512 train step、identity augmentationで比較した。

### メトリクスの解釈

予測数とGT数は194でcount accuracyは`1.0`だが、corner gateを通過したTPは26、FP/FNは各168である。受理ペア内のcenter誤差`0.445906 px`は3条件で最小だが、scale相対誤差`0.099535`とF1は最悪である。validationは1 epoch末の1点だけで、収束未了かどうかはcurveから確定できない。

### アーキテクチャ⇄メトリクスの因果考察

仮説として、LoRA・検出head・OBB headに加えて入力channel写像も同時最適化したため、512 stepでは事前学習backboneへ入る分布が動き、単純写像より最適化が不安定になった可能性がある。初期写像は同一なので、観測差は追加自由度と学習trajectoryによるものであり、adapter自体の上限性能を否定する結果ではない。

### 既存実験との比較

baselineの`repeat_rgb`に対してF1は`0.077319`低く、corner誤差は`1.202602 px`大きい。`red_only`に対してもF1とtotal lossで劣る。一方、受理ペアのcenter誤差のみ最良だった。

### 次に有効な実験

本方式を再評価する場合は、入力adapterだけ低いlearning rateに分離するか、最初のwarmup期間をfreezeしてから解放し、2048 step以上・複数seedで比較する。現状の512-stepプロトタイプ選定では優先しない。
