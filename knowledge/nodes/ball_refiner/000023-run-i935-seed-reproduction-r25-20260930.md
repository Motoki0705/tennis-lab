---
id: run-i935-seed-reproduction-r25-20260930
type: run
task: ball_refiner
sequence: 23
recorded_at: '2026-09-30'
title: 追加seedの事前10比較と固定共分散倍率による三seed診断
provider: codex
status: done
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
  cpu_seconds: 608.094179717009
  verified_input_files: 884
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
repro:
  commit: 9197995b
  branch: campaign930/i935-14-pipeline-candidate
  command: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-seed-reproduction-r25-20260930/compare_seeds.py
    --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-seed-reproduction-r25-20260930/plan.json
    --output /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-seed-reproduction-r25-20260930/results
---


事前宣言した[run24基準](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5909424600)では**FAIL（9/10）**。seed43は5/5、seed44は4/5で、人工gapの位置NLLだけがabsolute_12kより悪い（10.127395 > 9.992509 nat）。[判定と876ファイルのhash照合](../../runs/run-i935-seed-reproduction-r25-20260930/saved-metrics-gate.json)を記録した。通常入力23,007 frame、gap内observed 5,077 frameで、Meiji video_000の全camera/両halfを合算する。倍率適用前の判定であり、補正後の改善で救済しない。

[compare_seeds.py](../../runs/run-i935-seed-reproduction-r25-20260930/compare_seeds.py)は同じ教師・frame/PTS/gap・入力hashを照合し、既存bestが選択NLL最小（同点は早いepoch）であることを確認する。checkpointを選び直さず、全GMMをCPUで固定倍率1.8125148752087792により採点する。HDR MC2048/seed1729/levels50,90,95を維持し、mean/mixture/presenceの不変性も検査する。[全source/camera/half別の三seed表](../../runs/run-i935-seed-reproduction-r25-20260930/comparison.md)、[raw spread CSV](../../runs/run-i935-seed-reproduction-r25-20260930/results/raw-spread.csv)、[fixed spread CSV](../../runs/run-i935-seed-reproduction-r25-20260930/results/fixed-spread.csv)を保存した。各seed・min/max/spreadと母数を通常/gap別に示す。884入力のhashを処理前後に照合し、全配列からの集計が保存済みraw指標と一致した。既存bestはseed42/43/44のepoch41/45/27、absoluteはepoch46のまま。

これはseed42のcalibration halfでfitした単一倍率の固定転用であり、LOCO OOFでも独立test性能でもない。video_001、person/pose/courtは使わない。ユーザー判断Aの採用を自動撤回しないが、Bの再現確認は通過していない。実動画checkが通っても、既定切替にはこの負の結果について新しい判断が必要。

通常検証: [28テスト](../../runs/run-i935-seed-reproduction-r25-20260930/tests.log)成功（10条件それぞれの厳密不等式、非有限/未完分母、compiler cache再現、GMM較正・parity）。追加4Pythonファイルのruff/mypy成功。独立validatorは未指定・0回。

Meiji全体の固定倍率後observed NLLはseed42/43/44で6.420746 / 6.525769 / 6.514260 nat。rawからは全seedで改善するが、gap NLLは10.071598 / 9.873204 / 10.154584 natへ全seedで悪化する。補正後observed HDR90は0.899031 / 0.875907 / 0.897770、HDR95は0.924849 / 0.907637 / 0.927978。HDR90面積は約1.8倍、HDR50は0.616–0.671と過大被覆である。

全体の平均で局所的な過信を隠さない。calibration halfの補正後observed HDR90は0.842–0.863、HDR95は0.874–0.896に留まる。TrackNet observed NLLは全seedで補正後に悪化する。seed44 raw gap NLLの対absolute差はselection +0.085945 / calibration +0.182104 nat、cam0/1/2では+0.293353 / +0.067478 / +0.042460 natで、1cameraだけの例外ではない。因果的に学習seedのどの要因が悪化を生んだかは未同定。閾値や倍率を再fitして合格扱いにしない。

CPU専用の集計は608.09秒、GPU追加実行なし。[manifest](../../runs/run-i935-seed-reproduction-r25-20260930/results/manifest.json)が入力hashと選択checkpointを保持し、[出力hash](../../runs/run-i935-seed-reproduction-r25-20260930/output-sha256.json)で全診断を固定した。再計算時は出力先を新規directoryへ変更する。集計準備中の旧manifest seed metadata対応・lint/type修正前の失敗/中断ログはworktree outputs/campaign_930/i935/run25に残し、この最終集計とは混ぜない。
