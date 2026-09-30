---
id: run-i936-combined-overlap-cpu-r17-s936
type: run
task: ball_refiner_3d
sequence: 28
recorded_at: '2026-10-01'
title: 併用候補20kの固定重複窓CPU比較
provider: codex
status: done
issue: 936
date: '2026-10-01'
config: {primary_update: 20000, device: cpu, frames: 128, stride: 64, validation_rallies: 16}
metrics: {flow_overlap_rmse_m: 1.631605, flow_overlap_behind: 0}
artifacts:
  run_dir: knowledge/runs/run-i936-combined-overlap-cpu-r17-s936
parents: [run-i936-combined512-physics10-r16-s936, run-i936-overlap-cpu-r16-s936]
relations: []
papers: []
tags: []
---

[実行前登録](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5921223690)に従い、
run16(d)のprobe.pyを**byte単位でそのまま**使い、対象checkpointだけ(c)の両arm20kに変えた。
[plan](../../runs/run-i936-combined-overlap-cpu-r17-s936/plan.json)は35入力/code/checkpoint hashを固定。
CPU同士・同じ全ラリーnoiseで元窓と重複窓を比較し、全16val/6,383frame、4sample×8stepを保持。
[全表/可視camera層](../../runs/run-i936-combined-overlap-cpu-r17-s936/results/comparison.md)、
[診断と正式15軸の全判定](../../runs/run-i936-combined-overlap-cpu-r17-s936/results/rules.json)、
[粗さの全二乗和/p95](../../runs/run-i936-combined-overlap-cpu-r17-s936/results/roughness.json)、
[hash/資源/窓ownership](../../runs/run-i936-combined-overlap-cpu-r17-s936/results/manifest.json)を保存した。

flow平均のRMSE/gapは1.971/1.935→1.632/1.619m、behind4→0。
reprojection mean/p50/p95は20.722/12.108/64.076→21.209/12.142/63.636px。
full free accel/jerk p95は183.324/14560.393→165.414/11355.947。
全sampleもRMSE1.633m・behind0/70112で、平均だけの改善ではない。
回帰はRMSE2.055→1.701m・behind4→0だが、repro mean18.578→20.688px。

**seam診断はflow平均/全sample・回帰すべて不合格**。
flow平均はcamera2可視RMSE1.804→2.041m（+13.16%）、
元窓内free accel p95 +1.41%、jerk +8.91%が失敗。
全sampleでも同じ3軸（camera2 +13.13%、窓内accel+0.73%/jerk+6.46%）が失敗。
回帰はreprojection mean +11.36%、窓内accel+2.88%/jerk+16.37%が失敗。
seam free二乗和はflowで元の約0.02%/0.01%、全free二乗和も0.63%/0.35%まで下がったが、
窓内への移動を許さない事前規則を満たさない。blendの再調整・checkpoint選択はしない。

**正式規則も両arm未達**。RTS比で残る共通軸はfree accel/jerkとrepro mean/p50。
flow overlap平均はRTS比のfree accel1.046倍、free jerk1.592倍、repro mean1.290倍、p50 4.912倍。
3D RMSEやgap、all accel/jerkとrepro p95はRTSより小さい。behind=0はこの16valでの観測に限る。
flow固有優位も回帰overlap比event/camera2/3/repro3軸が悪く未達。
CPU元窓1.971mとCUDA(c)1.953mの乱数/数値差をblendの効果に数えない。

資源217.730秒、peak RSS976,998,400 bytes、最小空きRAM20,785,803,264 bytes。
GPU jobなし。全pin/元ownershipを実行前後で照合し、追加val/test配列0、全予測を保存。
推論module変更はなく、既存23tests（guard/粗さ/overlap、-n4）とruff/mypyが成功。
再現はprobe.pyを含む当bundleのcommitからplanの絶対pathを使ってCPU/native1で実行する。
学習validationのstride128、H/default、bank、#959 controlは据置。

残る最大の相対差は再投影p50。次の保存予測診断で多数の観測良好区間のずれと損失の関係を調べる。
