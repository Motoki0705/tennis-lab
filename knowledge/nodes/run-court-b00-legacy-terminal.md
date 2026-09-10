---
id: run-court-b00-legacy-terminal
type: run
title: B00：旧resolved-configの終了段階欠落で再生成preflight停止
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B00
  sfm_boundary_expansion_percent: 5.0
metrics:
  generated_frames: 0
repro:
  commit: ecf64a7aa6e21ed11e619cb75360ef44377aea8b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B00-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b00-legacy-terminal
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/court-b00-b03-generation
  log: knowledge/runs/run-court-b00-legacy-terminal/run.log
parents:
- run-b01-court-expansion-5
relations: []
tags:
- court
- synthetic-data
- canonical-generation
---

## 考察 / Findings

### 要約
現行alignmentへの再生成を開始したが、旧resolved-config.yamlにrequest.through_stageがなく、GPU推論・既存owner無効化前に停止した。

### アーキテクチャ詳細
最新runnerのalignment-scoped設定変更を使用。保存設定のinvocation cursor解析で明示エラーとなった。

### メトリクスの解釈
生成画像0。既存alignment/Court ownerは維持された。

### アーキテクチャ⇄メトリクスの因果考察
保存元のrun.jsonでreportがcompletedであることを確認し、scene writer lock下で欠落項目だけをreportとして補完した。元設定はoutputs/court-b00-b03-generation/before/B00/resolved-config.yamlへ保存。alignment証拠を合格扱いにする操作ではない。

### 既存実験との比較
B01はthrough_stageを持ち、正式生成成功。B00だけ旧invocation設定の移行が必要だった。

### 次に有効な実験
同じ生成設定でB00を再投入する。構成差分はB00-config-migration.jsonに記録した。
