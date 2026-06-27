---
id: run-i579-phase2
type: run
title: staged_phase2
issue: 579
provider: claude
session: 7fdebd68-e82c-43c1-b8f9-ded3802a3522
date: '2026-06-27'
status: done
config:
  model: dinov3_rope
  loss: default
  data: staged (t_max=1, tracknet+web)
metrics:
  f1: 0.034879
  loss: 0.00105
  mean_distance_px: 2.608411
  precision: 0.039573
  recall: 0.03118
repro:
  commit: 916575525cdaa7dcbf395d76246c0e4e800a870a
  branch: feat/issue-579-staged-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train_staged
    --config-name staged_phase2
artifacts:
  run_dir: knowledge/runs/run-i579-phase2
  log: .training_queue/logs/1782535208322054353_2561811_staged_phase2.log
  output_dir: outputs/ball_detection/staged/phase2/logs/run
  curves: knowledge/runs/run-i579-phase2/curves.png
  tb_logdir: outputs/ball_detection/staged/phase2/logs/run
parents:
- run-i579-phase1
relations: []
tags:
- ball_detection
- dinov3_rope
- staged
- t1
- web_mix
- phase2
---

## 考察 / Findings

### 要約
段階的学習 Phase 2（TrackNet+Web混合 / T=1 / Phase1 重みから継続 / 5ep）。test/f1=0.035（precision=0.040, recall=0.031）。Phase1 の退化（f1=0）から脱したが、絶対水準は i551 baseline(0.080) を下回る。**Web 混合＋継続学習が退化解からの脱出に寄与**した。

### アーキテクチャ詳細
Phase1 からの差分は **data.sources.web.enabled=true（Web 混合）** と `run.init_weights=phase1/last.ckpt`（継続学習）のみ。T=1・5ep・model=`dinov3_rope`・loss=default は同一。Web は `data/tennis/web/unified` を temporal sampling で混ぜる。

### メトリクスの解釈
test/loss=0.00105 と Phase1(0.00461) から大きく低下し、f1 が 0→0.035 に立ち上がった。mean_distance_px=2.61 はマッチ検出が出始めたことを示す。ただし precision/recall とも 3〜4% 台で実用水準には程遠い。

### アーキテクチャ⇄メトリクスの因果考察
Web データはアノテーション分布・シーン多様性が TrackNet と異なり、ボール画素のサンプルを増やすことで focal_bce の退化解を崩したと考えられる（仮説）。ただし起点が Phase1 の退化重みのため、5ep では立ち上がり途上に留まった。

### 既存実験との比較
Phase1 [[run-i579-phase1]]（f1=0.0）から明確に改善。i551 baseline [[run-i551-dinov3-rope-tracknet-train-retry1]]（T=1/20ep, f1=0.080）には未達で、合計10ep（Phase1+2）でも 20ep baseline に届かない。次フェーズ Phase 3 のマルチフレーム化で baseline を超える（[[run-i579-phase3]]）。

### 次に有効な実験
T=1 段階での epoch をさらに増やして web 混合の効果を飽和させてから多フレームへ移る。あるいは退化を避けるため Phase1 を skip し i551 baseline から直接 web 混合継続する。
