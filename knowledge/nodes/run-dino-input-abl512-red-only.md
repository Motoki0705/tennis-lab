---
id: run-dino-input-abl512-red-only
type: run
title: DINO入力 red channel only（512 step）
provider: codex
session: 01a06b47-7972-7521-a83c-be95cccf7d91
date: '2026-09-04'
status: done
config:
  model: dino-swin-l-4scale-lora-r8-red-only
  loss: detr-focal-aabb-giou-scale-axial
  data: procedural-court-800-identity-512
metrics:
  instance_precision: 0.221649
  instance_recall: 0.221649
  instance_f1: 0.221649
  instance_count_accuracy: 1.0
  instance_count_mae: 0.0
  matched_center_mean_error_px: 0.471567
  matched_scale_relative_error: 0.082448
  matched_axial_angle_mean_error_deg: 6.658131
  matched_corner_mean_error_px: 23.775142
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
    run.test_after_fit=true model.input_mode=red_only run.output_dir=court_alignment/dino_input_ablation_512_red_only
artifacts:
  run_dir: knowledge/runs/run-dino-input-abl512-red-only
  predictions: knowledge/runs/run-dino-input-abl512-red-only/pred_test.npz
  output_dir: outputs/court_alignment/dino_input_ablation_512_red_only/logs/version_0
  curves: knowledge/runs/run-dino-input-abl512-red-only/curves.png
  tb_logdir: outputs/court_alignment/dino_input_ablation_512_red_only/logs/version_0
parents: [run-dino-input-abl512-repeat-rgb]
relations: []
tags: [court-alignment, dino, detr, lora, red-only, input-ablation]
---

## 考察 / Findings

### 要約

ヒートマップをred channelだけへ入れ、green/blueを0にした条件は、512 step時点でF1 `0.221649`、validation loss `8.3421`、test loss `8.5969`となり、検出とtotal lossでは暫定首位だった。

### アーキテクチャ詳細

公式DINO Swin-L 4-scale、LoRA rank 8、1-class AABB + 長辺scale + axial orientation headを共通に用いた。入力のみ`[heatmap, 0, 0]`としてからImageNet正規化した。入力800×800、seed 42、train/val/test 512/128/128、identity augmentation、1 epochである。

### メトリクスの解釈

予測数とGT数は194でcount accuracyは`1.0`、32 px corner gateを通過したTPは43、FP/FNは各151だった。scale相対誤差`0.082448`は最良だが、受理ペアの軸角`6.658131 deg`とcorner`23.775142 px`は3条件中最大である。よってF1首位だけで幾何品質も首位とは判断できない。

### アーキテクチャ⇄メトリクスの因果考察

仮説として、channel非対称性がImageNet RGB filterへ明確な信号差を与え、短期のclassification/scale最適化を助けた可能性がある。一方でgreen/blueが常に0である大きな分布ずれが、長辺軸の精密化には不利だった可能性がある。1 seed・1 epochなので因果は未確定である。

### 既存実験との比較

`repeat_rgb`よりF1は`0.010309`、scale相対誤差は`0.007446`良い。一方、軸角は`1.551200 deg`、cornerは`2.884465 px`悪い。`learnable_1x1`よりF1とlossは明確に良い。

### 次に有効な実験

検出F1を主指標にする場合の暫定候補として、`repeat_rgb`と同一の延長予算・複数seedで再比較する。幾何alignmentを最終目的とするため、選定基準にはF1だけでなくcorner誤差と角度誤差を併記する。
