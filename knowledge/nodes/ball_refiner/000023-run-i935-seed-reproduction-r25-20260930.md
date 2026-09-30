---
id: run-i935-seed-reproduction-r25-20260930
type: run
task: ball_refiner
sequence: 23
recorded_at: '2026-09-30'
title: 追加seedの事前10比較と固定共分散倍率による三seed診断
provider: codex
status: running
config:
  seeds:
  - 42
  - 43
  - 44
  covariance_multiplier: 1.8125148752087792
  refit: false
metrics:
  reproduction_comparisons_passed: 9
  reproduction_comparisons_total: 10
artifacts:
  run_dir: knowledge/runs/run-i935-seed-reproduction-r25-20260930
parents:
- run-i935-seed44-retry-r24-20260930
relations:
- to: run-i935-precision-variants-s42-r21-20260930
  rel: compares
papers: []
tags: []
issue: 935
date: '2026-09-30'
session: 01a0f263-c4b3-7972-a86a-90c1f4c56cc8
---

事前宣言した[run24基準](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5909424600)では**FAIL（9/10）**。seed43は5/5、seed44は4/5で、人工gapの位置NLLだけがabsolute_12kより悪い（10.127395 > 9.992509 nat）。[判定と876ファイルのhash照合](../../runs/run-i935-seed-reproduction-r25-20260930/saved-metrics-gate.json)を記録した。通常入力23,007 frame、gap内observed 5,077 frameで、Meiji video_000の全camera/両halfを合算する。倍率適用前の判定であり、補正後の改善で救済しない。

[compare_seeds.py](../../runs/run-i935-seed-reproduction-r25-20260930/compare_seeds.py)は同じ教師・frame/PTS/gap・入力hashを照合し、既存bestが選択NLL最小（同点は早いepoch）であることを確認する。checkpointを選び直さず、全GMMをCPUで固定倍率1.8125148752087792により採点する。HDR MC2048/seed1729/levels50,90,95を維持し、mean/mixture/presenceの不変性も検査する。全source/camera/halfの三seed spreadは集計中。

これはseed42のcalibration halfでfitした単一倍率の固定転用であり、LOCO OOFでも独立test性能でもない。video_001、person/pose/courtは使わない。ユーザー判断Aの採用を自動撤回しないが、Bの再現確認は通過していない。実動画checkが通っても、既定切替にはこの負の結果について新しい判断が必要。

通常検証: [28テスト](../../runs/run-i935-seed-reproduction-r25-20260930/tests.log)成功（10条件それぞれの厳密不等式、非有限/未完分母、compiler cache再現、GMM較正・parity）。追加4Pythonファイルのruff/mypy成功。独立validatorは未指定・0回。
