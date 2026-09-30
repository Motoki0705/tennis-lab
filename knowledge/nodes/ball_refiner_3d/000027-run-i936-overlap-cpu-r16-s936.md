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
status: done
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


## CPU結果: flowはseam診断成功、正式優位は未達

source **61327fa4002cea4ff5f460dc59fb1207bacc06a0**、tracked diffなし、実験試行は1回、GPU0件。
[全表](../../runs/run-i936-overlap-cpu-r16-s936/results/comparison.md)、[全診断/正式規則](../../runs/run-i936-overlap-cpu-r16-s936/results/rules.json)、
[粗さ分解](../../runs/run-i936-overlap-cpu-r16-s936/results/roughness.json)、[hash/資源/窓](../../runs/run-i936-overlap-cpu-r16-s936/results/manifest.json)を保存した。
全16val/6,383frame、全checkpoint/input/code hash不変、元ownershipも一致。各ラリーにnoiseと両方式の全sampleを保存した。

| CPU方式/20k | RMSE / gap m | free accel / jerk p95 | repro mean / p50 / p95 px | behind |
|---|---|---|---|---|
| flow元窓/平均 | 2.033 / 2.091 | 1951.048 / 202133.910 | 21.865 / 11.548 / 74.626 | 4/17528 |
| flow overlap/平均 | 1.728 / 1.972 | 1665.657 / 163240.337 | 22.241 / 11.368 / 75.116 | 0/17528 |
| flow overlap/全sample | 1.730 / 1.973 | 1666.438 / 162661.367 | 22.246 / 11.382 / 74.882 | 0/70112 |
| 回帰元窓 | 2.412 / 2.311 | 2065.433 / 216758.040 | 21.062 / 11.484 / 72.857 | 4/17528 |
| 回帰 overlap | 1.968 / 2.127 | 1749.900 / 182237.811 | 21.754 / 11.319 / 75.889 | 0/17528 |

flowは**平均・全sampleともseam診断成功**。元seamでのfree accel/jerk二乗和は平均で対照の1.05%/1.47%、
全free二乗和は51.56%/52.52%。full/元窓内free p95もすべて非悪化、RMSE8軸・再投影3軸は105%以下。
behind4→0（全sample16→0）。しかし窓内free accel/jerk p95の低下は約0.7%/0.3%で、窓内の粗さは残る。
これは固定dev上の結果で、他入力でbehindゼロを保証する方式ではない。

回帰はseam二乗和を1.11%/1.58%、全free二乗和を53.26%/54.86%へ下げ、behind4→0。
ただしcamera2 RMSEが105%を超えるため**診断不合格**。両arm共通改善とは呼ばない。

**正式規則は両armとも不合格。** flow平均/全sample・回帰の再投影mean/p50が混合平均より悪く、
RTSには加速度/jerk all/freeおよび再投影mean/p50/p95で非悪化を満たさない。
flowは回帰overlapに対する再投影mean/p50が悪いためdiffusion固有の正式優位もない。
正depthの分母は元窓17524→overlap17528（全sample70096→70112）となり、behindを捨てた改善ではない。

CPU control flow RMSE2.033mと(a) CUDA2.018mは同一ではない。乱数/数値計算deviceが異なるため、
改善量は必ず同CPU controlから測った。文脈の重なりと固定blendの併用であり、純粋なseam因果分解ではない。
(c)の学習・validationは元stride128のまま保持し、この結果を見てoverlapを混ぜない。
次の候補として(c)の結果回収後に固定overlap評価を提案できるが、本runでは追加検証しない。

実測 **150.799秒、peak RSS979,320,832 bytes、最小空きRAM27,492,020,224 bytes**。
実行log・全予測・hashを保持する。学習・TensorBoardなし、test/追加48val・Meiji/pipelineも0。
再現はsource61327fa4のcheckoutで[execution-context.json](../../runs/run-i936-overlap-cpu-r16-s936/execution-context.json)のcommandを実行する。
results先が既存なら拒否し、実データ比較の自動再試行はしない。
