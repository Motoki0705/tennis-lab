---
id: run-dino-input-abl512-repeat-rgb
type: run
title: DINO入力 repeat RGB（512 step）
provider: codex
session: 01a06b47-7972-7521-a83c-be95cccf7d91
date: '2026-09-04'
status: done
config:
  model: dino-swin-l-4scale-lora-r8-repeat-rgb
  loss: detr-focal-aabb-giou-scale-axial
  data: procedural-court-800-identity-512
metrics:
  instance_precision: 0.21134
  instance_recall: 0.21134
  instance_f1: 0.21134
  instance_count_accuracy: 1.0
  instance_count_mae: 0.0
  matched_center_mean_error_px: 0.514214
  matched_scale_relative_error: 0.089894
  matched_axial_angle_mean_error_deg: 5.106931
  matched_corner_mean_error_px: 20.890677
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
    run.test_after_fit=true model.input_mode=repeat_rgb run.output_dir=court_alignment/dino_input_ablation_512_repeat_rgb
artifacts:
  run_dir: knowledge/runs/run-dino-input-abl512-repeat-rgb
  predictions: knowledge/runs/run-dino-input-abl512-repeat-rgb/pred_test.npz
  output_dir: outputs/court_alignment/dino_input_ablation_512_repeat_rgb/logs/version_0
  curves: knowledge/runs/run-dino-input-abl512-repeat-rgb/curves.png
  tb_logdir: outputs/court_alignment/dino_input_ablation_512_repeat_rgb/logs/version_0
parents: []
relations:
- {to: run-dino-input-abl512-learnable-1x1, rel: compares}
- {to: run-dino-input-abl512-red-only, rel: compares}
tags: [court-alignment, dino, detr, lora, repeat-rgb, input-ablation]
---

## 考察 / Findings

### 要約

1chヒートマップの単純RGB複製は、512 step時点で F1 `0.211340`。3条件中、軸角誤差 `5.106931 deg` とcorner誤差 `20.890677 px` が最良で、短期学習の幾何精度ベースラインとして有効だった。

### アーキテクチャ詳細

公式DINO Swin-L 4-scale COCO checkpointをstrict load後、1-class分類headと長辺scale・`cos(2θ), sin(2θ)`の軸方向headへ拡張した。LoRA rank 8をattention/FFNへ適用し、入力は同一heatmapをRGB各channelへ複製後にImageNet正規化した。入力は800×800、augmentationはidentity、seed 42、train/val/testは512/128/128、1 epochである。

### メトリクスの解釈

予測数とGT数はどちらも194でcount accuracyは`1.0`だが、32 px corner gateを通過したTPは41、FP/FNは各153である。したがってcount正解はinstance幾何の正解を意味しない。center・scale・angle・corner値は受理された41組に対する値である。validationは1点のみのため、生成済みcurveから収束は判定できない。

### アーキテクチャ⇄メトリクスの因果考察

仮説として、3channelすべてに同一の線強度を与えることで、ImageNet事前学習backboneのchannel間コントラストは失われる一方、初段filterへ均一なedge evidenceが入り、短期でも安定して角度・外形を学習できた。中心誤差が小さいのにcorner gateのTPが少ないことから、残課題は主に長辺scaleと軸方向の組合せにある。

### 既存実験との比較

`red_only`よりF1は`0.010309`低いが、軸角は`1.551200 deg`、cornerは`2.884465 px`良い。`learnable_1x1`よりF1、scale、angle、cornerが良く、短期予算では追加adapterより単純複製が安定した。

### 次に有効な実験

`repeat_rgb`と`red_only`を優先し、複数seedか2048 step以上へ延長してF1とcorner誤差の順位が維持されるか確認する。あわせてscale/axis lossのvalidation推移を複数epochで観測する。
