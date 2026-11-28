---
id: run-i2759-v4-hitter-rule-eval
type: run
task: tennis_scene
sequence: 110
recorded_at: '2026-11-29'
title: V4のルールベース打者推定と独立映像監査
issue: 2759
provider: codex
date: '2026-11-29'
status: done
summary_impact: updated
config:
  benchmark: meiji_4cam/multiview-v4
  rule_version: court-hitter-v1.217835f4f9d0
  split_clips: {train: 47, val: 11, test: 11}
metrics:
  val_shots: 60
  val_evaluable: 60
  val_correct_resolved: 59
  val_false_resolved: 0
  val_coverage: 0.9833333333333333
  test_shots: 51
  test_evaluable: 51
  test_correct_resolved: 49
  test_false_resolved: 0
  test_coverage: 0.9607843137254902
repro:
  commit: 2f16173d56864c88f5d4b580778d35135d411811
  command: >-
    .venv/bin/python -m src.tennis_scene.analytics.hitter_eval
    --benchmark data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4
    --manifest-sha256 "$(jq -r .manifest_sha256 knowledge/runs/run-i2759-v4-hitter-rule-eval/visual_audit.json)"
    evaluate --ground-truth knowledge/runs/run-i2759-v4-hitter-rule-eval/visual_ground_truth.csv
    --split test
artifacts:
  run_dir: knowledge/runs/run-i2759-v4-hitter-rule-eval
  ground_truth: knowledge/runs/run-i2759-v4-hitter-rule-eval/visual_ground_truth.csv
  visual_audit: knowledge/runs/run-i2759-v4-hitter-rule-eval/visual_audit.json
  validation: knowledge/runs/run-i2759-v4-hitter-rule-eval/val_evaluation.json
  evaluation: knowledge/runs/run-i2759-v4-hitter-rule-eval/test_evaluation.json
parents:
- run-i2733-meiji-event-labels-v4
relations: []
papers: []
tags:
- meiji_4cam
- shot_hitter
- multiview_v4
---

V4の注釈時刻・コート側と3Dの球・手首・rootで打者候補を判定した。モデルの予測を閲覧する前に22クリップの111件を4視点で確認し、ローカルIDを固定した。確定したval 59件とtest 49件はいずれもGTに一致した。3件の未確定では打者のtrackが接触窓に欠けていた。

V4の固定split内で、打者推定headと比較できるベースラインを追加した。3分割は同一試合のため、評価の適用範囲はその映像条件とV4入力に限られる。 [判定根拠・再現手順](../../runs/run-i2759-v4-hitter-rule-eval/README.md)
