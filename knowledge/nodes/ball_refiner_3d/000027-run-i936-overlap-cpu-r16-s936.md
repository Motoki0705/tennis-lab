---
id: run-i936-overlap-cpu-r16-s936
type: run
task: ball_refiner_3d
sequence: 27
recorded_at: '2026-10-01'
title: 固定512train重みのCPU三角重複窓比較
issue: 936
provider: codex
date: '2026-10-01'
status: planned
config:
  primary_update: 20000
  frames: 128
  stride_control: 128
  stride_candidate: 64
  samples: 4
  steps: 8
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i936-overlap-cpu-r16-s936
parents:
- run-i936-pilot512-t128-r15-s936
- run-i936-roughness-r14-s936
relations: []
papers: []
tags: []
session: 01a0f3ed-4151-7303-886f-52e91fdac10e
---

[実行前の理由・規則](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5918885662)を正本とする。
[plan.json](../../runs/run-i936-overlap-cpu-r16-s936/plan.json)は同じ512train/20k checkpointのhash、全code、固定CPU比較を保持する。
run14でseamがfree加速度二乗和の41〜42%を占めたため、学習を増やさず重複窓による改善を測る。
窓内粗さも残るので、seamだけを測って成功とはしない。

元T128/stride128をCPUで再推論し、T128/stride64の三角重みblendと同じ全ラリーnoiseで比較する。
CPU RNG/numericsとCUDAの差をblend効果に含めない。窓startは既存helperのmin(start,T-3)、右padding/絶対時刻を維持。
全sampleを別々にblendし、GT/eventで重みを変えない。全frame・元16valを使い、追加val/test配列は読まない。
判定はseam二乗和半減、full/元窓内free粗さの非悪化、誤差105%以下、behind非増加。
正式15軸＋behindゼロ規則を別途適用。結果を見てblend/stride/checkpointを選び直さない。

CPU1 process/native1、18分上限、見積RSS2GB/出力50MB、空きRAM6GiB下限、GPU0件。
モデル凍結・新規学習なし。TensorBoardは適用せず、予測NPZ・比較表・全軸・hash・資源を保存する。
通常検証の初回は解析fixtureがT16以上を要求するため6件失敗した。fixture寸法を直し、短い系列・末尾・sample/time保持・
教師非依存・既知seamの改善を9testsで確認。既存contextの2testsは初回から成功。ruff/mypyを併用する。
