---
id: run-i579-phase4
type: run
title: staged_phase4_resume
issue: 579
provider: codex
session: 019f07f0-adfc-7391-820a-97f5f255c67b
date: '2026-06-27'
status: done
config:
  model: dinov3_rope
  loss: default
  data: staged (t_max=8, t1_prob=0.5, tracknet+web)
metrics:
  f1: 0.048286
  loss: 0.001104
  mean_distance_px: 2.635513
  precision: 0.048228
  recall: 0.048344
repro:
  commit: 916575525cdaa7dcbf395d76246c0e4e800a870a
  branch: feat/issue-579-staged-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train_staged
    --config-name staged_phase4 run.resume=outputs/ball_detection/staged/phase4/logs/run/checkpoints/last.ckpt
    run.init_weights=null
artifacts:
  run_dir: knowledge/runs/run-i579-phase4
  log: .training_queue/logs/1782544799619240312_13706_staged_phase4_resume.log
  output_dir: outputs/ball_detection/staged/phase4/logs/run
  curves: knowledge/runs/run-i579-phase4/curves.png
  tb_logdir: outputs/ball_detection/staged/phase4/logs/run
parents:
- run-i579-phase3
relations:
- to: run-i579-phase3
  rel: contradicts
tags:
- ball_detection
- dinov3_rope
- staged
- multiframe
- web_mix
- phase4
---

## 考察 / Findings

### 要約
段階的学習 Phase 4（TrackNet+Web混合 / マルチフレーム T∈[1,8] / Phase3 重みから継続 / 10ep）。test/f1=0.048（precision=recall≈0.048）で、最良の Phase3(0.106) から**約半分に悪化（回帰）**。**マルチフレーム段階での Web 混合は逆効果**という結論。なお初回 phase4 run は DataLoader worker が Terminated（メモリ起因と推定）で失敗し、本runは last.ckpt から resume して完走（provider=codex）。

### アーキテクチャ詳細
Phase3 からの差分は **data.sources.web.enabled=true（Web 混合）** のみ（t_max=8・t1_prob=0.5・10ep・model/loss は同一）、`init_weights=phase3/last.ckpt`。本 run は `run.resume=phase4/last.ckpt run.init_weights=null` で中断した phase4 を継続したもので、phase4 config 自体は同一。

### メトリクスの解釈
test/f1=0.048・loss=0.00110 で、Phase3（f1=0.106, loss=0.000813）より f1 低下・loss 上昇。mean_distance_px=2.64 も Phase3(2.53) より悪化。precision と recall がほぼ同値で、全体的に検出品質が後退している。

### アーキテクチャ⇄メトリクスの因果考察
T=1 段階では退化脱出に効いた Web 混合が、マルチフレーム段階では悪化要因になった。Web データの temporal sampling が TrackNet ほど一貫した時系列ラベルを持たず、T∈[2,8] バッチでノイズの多い時間ターゲットを注入し、Phase3 で得た多フレーム表現を劣化させた可能性が高い（仮説）。crash→resume による optimizer/スケジューラ状態の不連続も僅かに寄与しうる（仮説）。

### 既存実験との比較
Phase3 [[run-i579-phase3]]（TrackNet単独マルチフレーム, f1=0.106）に対し明確な回帰。Web 混合は Phase1→Phase2（T=1）では +0.035 と有効だったのに対し、Phase3→Phase4（マルチフレーム）では -0.058 と符号が反転し、混合の効果が段階依存であることを示す。

### 次に有効な実験
マルチフレーム段階では Web 混合を外し、Phase3 recipe を最終 recipe として epoch増・backbone部分解凍で深掘りする。Web を活かすなら temporal ラベル品質を検証・フィルタしてから単一フレーム段階に限定して混合する。
