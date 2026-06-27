---
id: run-i579-phase1
type: run
title: staged_phase1
issue: 579
provider: claude
session: 7fdebd68-e82c-43c1-b8f9-ded3802a3522
date: '2026-06-27'
status: done
config:
  model: dinov3_rope
  loss: default
  data: staged (t_max=1, tracknet only)
metrics:
  f1: 0.0
  loss: 0.004611
  mean_distance_px: 0.0
  precision: 0.0
  recall: 0.0
repro:
  commit: 916575525cdaa7dcbf395d76246c0e4e800a870a
  branch: feat/issue-579-staged-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train_staged
    --config-name staged_phase1
artifacts:
  run_dir: knowledge/runs/run-i579-phase1
  log: .training_queue/logs/1782535208299772497_2561795_staged_phase1.log
  output_dir: outputs/ball_detection/staged/phase1/logs/run
  curves: knowledge/runs/run-i579-phase1/curves.png
  tb_logdir: outputs/ball_detection/staged/phase1/logs/run
parents:
- run-i551-dinov3-rope-tracknet-train-retry1
relations: []
tags:
- ball_detection
- dinov3_rope
- staged
- t1
- tracknet
- phase1
---

## 考察 / Findings

### 要約
段階的学習 Phase 1（TrackNet単独 / T=1 / scratch / 5ep）。test/f1=0.0、precision=recall=0.0 と**完全に退化**し、閾値を超える peak を一切出さない（mean_distance_px=0 は検出ゼロの副作用）。scratch から 5 epoch では `dinov3_rope` 検出器は学習が立ち上がらない。

### アーキテクチャ詳細
model=`dinov3_rope`（DINOv3 ViT-B/16 frozen backbone + 4層/8head/dim256 decoder に (time,y,x) 3軸 RoPE）、loss=default(focal_bce)、data=`staged`（t_max=1 なので単一フレーム、tracknet のみ・web無効）。manual-optimization の variable-T runner を T=1 に縮退させた形。`run.init_weights=null` で scratch 起動、max_epochs=5、EBS=8。i551 baseline（同一 model を rgb_sequence で 20ep 学習）との差は **scratch・5ep・staged datamodule** の3点。

### メトリクスの解釈
test/loss=0.00461 は他フェーズ（0.0008〜0.0011）より一桁高く、val/f1 も学習中ずっと 0。検出ヘッドが全画素を低 logit に潰した退化解に落ちている。mean_distance_px=0.0 は「マッチした検出が存在しない」ことの帰結で、距離が良いわけではない。

### アーキテクチャ⇄メトリクスの因果考察
i551 では同一 backbone・decoder で 20ep かけて f1=0.080 まで到達していた。scratch + 5ep では focal_bce 下の極端なクラス不均衡（ボール画素はごく少数）を脱せず、「何も出さない」局所解に留まったと考えられる（仮説）。warmup_epochs=1 で実効学習は4ep相当しかない点も寄与した可能性がある（仮説）。

### 既存実験との比較
i551 baseline [[run-i551-dinov3-rope-tracknet-train-retry1]]（T=1/20ep, f1=0.080）に対し、本 run は同系統だが 5ep scratch で f1=0.0。段階的スケジュールの起点としては失敗だが、後続 Phase 2 で web 混合＋継続学習により退化を脱する（[[run-i579-phase2]]）。

### 次に有効な実験
起点を scratch ではなく i551 baseline ckpt から継続するか、Phase 1 の epoch を増やす（10〜20ep）。または warmup 短縮・focal_bce の pos_weight 調整で立ち上がりを早める。
