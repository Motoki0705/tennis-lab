---
id: run-court-b02-narrow-fov
type: run
title: B02：v14再生成と画角の事前適合性
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B02
  hfov_degrees:
  - 45.0
  - 90.0
  sfm_boundary_expansion_percent: 5.0
metrics:
  pre_render_valid: 1678
  proposals: 2232
repro:
  commit: ecf64a7aa6e21ed11e619cb75360ef44377aea8b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B02-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b02-narrow-fov
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/court-b00-b03-generation
  log: knowledge/runs/run-court-b02-narrow-fov/run.log
parents:
- run-court-b01-canonical-sfm
relations: []
tags:
- court
- synthetic-data
- fov
- alignment
---

## 考察 / Findings

### 要約
v14 alignmentは1面でfit/holdout採用を通過したが、45〜90度の画角では必要な4点が映る候補が1,678/2,232しかなく、minimum_accepted_frames=2000を満たせずdataset公開は停止した。75〜110度への変更で事前有効候補2,128まで改善することをCPU投影比較で確認し、alignmentを再利用するCourt-only再生成を投入した。

### アーキテクチャ詳細
正式ScenePipelineRunnerとpublic NHT rendererを使用。半径・形状・高さ・SfM外周許容は維持。

### メトリクスの解釈
事前有効候補はambiguous near/farを除き、全対象コートで少なくとも4点が画像内へ投影される候補。描画後の可視性や下流精度を保証する数ではない。

### アーキテクチャ⇄メトリクスの因果考察
狭いSfM水平範囲でコート中心を向くカメラでは画角が狭いと十分なコート点が映らない。画角を広げることで境界制約を緩めず可視点数を増やせる。最低枚数・採用率のgateは変更していない。

### 既存実験との比較
B01は3面を含むため45〜90度で正式公開できた。同じ設定が単一コートに十分とは限らなかった。

### 次に有効な実験
75〜110度の画角で正式生成・再読込・SfM包含を検証する。
