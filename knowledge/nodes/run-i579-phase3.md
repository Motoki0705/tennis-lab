---
id: run-i579-phase3
type: run
title: staged_phase3
issue: 579
provider: claude
session: 7fdebd68-e82c-43c1-b8f9-ded3802a3522
date: '2026-06-27'
status: done
config:
  model: dinov3_rope
  loss: default
  data: staged (t_max=8, t1_prob=0.5, tracknet only)
metrics:
  f1: 0.106227
  loss: 0.000813
  mean_distance_px: 2.532932
  precision: 0.099315
  recall: 0.114173
repro:
  commit: 916575525cdaa7dcbf395d76246c0e4e800a870a
  branch: feat/issue-579-staged-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train_staged
    --config-name staged_phase3
artifacts:
  run_dir: knowledge/runs/run-i579-phase3
  log: .training_queue/logs/1782535208343333873_2561827_staged_phase3.log
  output_dir: outputs/ball_detection/staged/phase3/logs/run
  curves: knowledge/runs/run-i579-phase3/curves.png
  tb_logdir: outputs/ball_detection/staged/phase3/logs/run
parents:
- run-i579-phase2
relations:
- to: run-i551-dinov3-rope-tracknet-t3-train
  rel: compares
tags:
- ball_detection
- dinov3_rope
- staged
- multiframe
- tracknet
- phase3
---

## 考察 / Findings

### 要約
段階的学習 Phase 3（TrackNet単独 / マルチフレーム T∈[1,8] / Phase2 重みから継続 / 10ep）。test/f1=0.106（precision=0.099, recall=0.114）で**本Issue4フェーズ中の最良**、かつ i551 の T=1(0.080)/T=3(0.094) baseline をいずれも上回る。マルチフレーム文脈＋epoch増が最大の寄与要因。

### アーキテクチャ詳細
Phase2 からの差分は **data.t_max=1→8（variable-T サンプリング、t1_prob=0.5）**、**web 無効化（TrackNet単独に戻す）**、**max_epochs 5→10**、`init_weights=phase2/last.ckpt`。バッチ単位で T をサンプリングし、T依存の物理バッチ `batch_size_by_t={1:8,2:6,...,8:2}` と accumulate で EBS≈8 を揃える manual-optimization runner。

### メトリクスの解釈
test/loss=0.000813 と全フェーズ最小、f1=0.106 で Phase2(0.035) から3倍。mean_distance_px=2.53 も最小で局在精度も僅かに改善。マルチフレーム化と epoch 倍増が同時に効いている。

### アーキテクチャ⇄メトリクスの因果考察
時系列文脈（隣接フレーム）により動くボールの peak が立てやすくなり、frozen backbone でも decoder が時間方向の手がかりを使えたと考えられる（仮説）。ただし T 拡張・web除去・epoch倍増が同時変更のため、各軸の寄与は本runだけでは分離不能（交絡）。

### 既存実験との比較
i551 [[run-i551-dinov3-rope-tracknet-t3-train]]（rgb_sequence, num_frames=3 固定, 20ep, f1=0.094）を、より少ない総epochかつ variable-T で上回った。variable-T（T∈[1,8]）が固定T=3より有利な可能性を示唆（仮説）。後続 Phase 4 で web 混合を足すと f1=0.048 へ**悪化**する（[[run-i579-phase4]]）。

### 次に有効な実験
本構成（TrackNet単独・variable-T）を最良 recipe として、epoch増・backbone部分解凍で絶対水準（f1≈0.1）を引き上げる。web 混合は Phase4 で逆効果のため多フレーム段階では加えない。
