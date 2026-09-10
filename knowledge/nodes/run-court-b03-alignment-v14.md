---
id: run-court-b03-alignment-v14
type: run
title: B03：v14再生成と画角の事前適合性
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B03
  hfov_degrees:
  - 45.0
  - 90.0
  sfm_boundary_expansion_percent: 5.0
metrics:
  alignment_published: true
  accepted_courts: 1
repro:
  commit: ad5554ab0cfdc433408b5762f3eb8792a30e6698
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B03-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b03-alignment-v14
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/court-b00-b03-generation
  log: knowledge/runs/run-court-b03-alignment-v14/run.log
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
v14 alignmentを1面で正常公開。旧配置を使った事前投影比較で45〜90度は1,714/2,256のみ有効、75〜110度なら2,142有効だった。新alignmentのstage completedを確認してから旧画角の実行を共有queue経由で意図的に停止した。これはalignment計算失敗ではなく、採用枚数不足が予想される描画を避けるため。次のCourt-only実行が中断stageを正式に回復し、更新済alignmentを保持して続行する。

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
