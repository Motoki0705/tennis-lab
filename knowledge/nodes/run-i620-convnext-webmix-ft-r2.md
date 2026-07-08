---
id: run-i620-convnext-webmix-ft-r2
type: run
title: i620_convnext_webmix_ft_r2
issue: 620
provider: claude
date: '2026-07-08'
status: done
config:
  model: conv_next_unet
metrics:
  precision: 0.277183
  recall: 0.332018
  f1: 0.302132
  mean_distance_px: 2.538643
repro:
  commit: 72230f2db31f9f6a99cfe6d2bdc24824d3f4d57e
  branch: feat/issue-620-web-mix-datamodule
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv/bin/python -m
    src.tasks.ball_detection.scripts.train_staged model=conv_next_unet data.t_max=8
    data.t_distribution=fixed data.val_num_frames=8 data.heatmap_size='[72,128]' data.num_workers=2
    data.sources.tracknet.enabled=true data.sources.tracknet.sample_stride=4 data.sources.web.enabled=true
    'data.sources.web.splits=[train]' 'data.sources.web.sources=[ball_yolo,kaggle_backview]'
    data.sources.web.sampling.mode=temporal data.sources.web.sampling.temporal.frame_step=1
    data.sources.web.sampling.temporal.sample_stride=2 data.sources.web.sampling.temporal.max_frame_gap=1
    training.learning_rate=5e-5 training.trainer.max_epochs=16 training.checkpoint.monitor=val/f1
    training.checkpoint.mode=max training.early_stopping.enabled=true training.early_stopping.monitor=val/f1
    training.early_stopping.mode=max training.early_stopping.patience=6 training.early_stopping.min_delta=0.0005
    'run.init_weights="/home/kamimura/projects/tennis-lab/outputs/ball_detection/convnext_v2_scratch/logs/version_0/checkpoints/ball-detection-epoch=25.ckpt"'
    run.output_dir=/home/kamimura/projects/tennis-lab/outputs/ball_detection/convnext_mdd_webmix_ft
artifacts:
  run_dir: knowledge/runs/run-i620-convnext-webmix-ft-r2
  predictions: knowledge/runs/run-i620-convnext-webmix-ft-r2/pred_test.npz
  log: .training_queue/logs/1783465025685952314_306728_i620_convnext_webmix_ft_r2.log
  output_dir: outputs/ball_detection/convnext_mdd_webmix_ft/logs/run
  curves: knowledge/runs/run-i620-convnext-webmix-ft-r2/curves.png
  tb_logdir: outputs/ball_detection/convnext_mdd_webmix_ft/logs/run
parents:
- run-i618-convnext-v2-scratch
relations:
- to: run-i618-convnext-v2-scratch-ft
  rel: compares
tags:
- ball_detection
- conv_next_unet
- web-data
- negative-result
---

## 考察 / Findings

### 要約
scratch-e25 起点の TrackNet+web 混合 fine-tune（fixed T=8, web=ball_yolo+kaggle_backview
train-only 混合 62.3%）は**崩壊**した: val/f1(純TrackNet) は epoch 0 で 0.347、train/f1 は
全期間 0.17 で平坦、ES ep6 停止。GPT-5.4 診断（.codex-runs/diag_webmix_collapse_report.md）
により**データバグは高確度で否定**され、主因は**無重み混合によるドメイン競合**
（+ 比較上のメトリクス差の増幅）。convnext+mdd 経路での web 混合ラインはここで閉じる。

### アーキテクチャ詳細
[[run-i618-convnext-v2-scratch]] と同一モデル。データのみ staged datamodule の新機能
（splits 制御 + fixed-T、PR #624）で web temporal T=8 gapless 窓（max_frame_gap=1,
sample_stride=2, 計 6,481 窓）を train にのみ混合。lr 5e-5, EBS 8, init_weights=scratch-e25。

### メトリクスの解釈
frontmatter は last ckpt の test 値。**注意: この run は metrics.nms_kernel=9（default）で
走っており、初期 ckpt の 0.7198 (nms=3+subpixel) と直接比較できない**（発注ミス、診断で発覚）。
ただし run 内相対でも改善ゼロ + train/f1 0.17 平坦なので崩壊自体は確実。

### アーキテクチャ⇄メトリクスの因果考察
診断の証拠: (1) web T=8 教師信号のアラインメントは tracknet と同等（argmax↔GT 平均 1.5px、
gapless 100%、mdd 統計も正常）。(2) 初期 ckpt は web フレームで有効ピーク 0（tracknet では
score 0.936）— web は「正しいが別分布」で、optimizer step の 62.3% が web 勾配になり
TrackNet 表現が 1 epoch で破壊された。(3) ball_yolo は全正例・kaggle_backview は裏側視点で
可視率 0.334 — いずれも放送ドメインから遠い。放送類似の roboflow 静止画は non-temporal で
T=8 mdd 窓を構成できないため、**convnext+mdd で「放送頑健性のための web 混合」は構造的に
成立しにくい**。

### 既存実験との比較
- [[run-i618-convnext-v2-scratch-ft]]（同起点・TrackNet-only 低lr ft）: 改善なしで停滞。
  本 run（web 混合）: 破壊。→ scratch-e25 への追加学習は両アームとも負。
- 実クリップ頑健性は学習でなく trajectory gate（PR #622、誤棄却 0/811 で
  テレポート除去）で担保する方が確実と判明。

### 次に有効な実験
- web データの価値回収は可変 T の DINOv3 路線（Colab: train_ball_dinov3_staged.sh、
  roboflow 静止画も T=1 で使える staged schedule）で行う。
- convnext で再挑戦するなら: source 重み付きサンプリング（tracknet≥0.7）+
  source 別 train metrics + nms_kernel=3 でのメトリクス統一が前提
  （診断レポート優先度 A/C/D）。
