---
id: run-i936-condition-readout-r11-s936
type: run
task: ball_refiner_3d
sequence: 17
recorded_at: '2026-09-30'
title: 凍結条件tokenの平均read-outはval 7.49mm：大きな情報欠落を支持せず
issue: 936
provider: codex
status: done
config:
  encoder: fixed final flow at 20000 updates
  target: all-component mixture mean, no GT in fit
  fit_split: all 64 train rallies
  evaluation_split: all 16 val rallies
  solver: float64 gelsd, rcond=1e-12, intercept, no ridge
  diagnostic_tolerance_m: 0.1
  native_threads: 1
metrics:
  train_rmse_m: 0.0016491827595806771
  val_rmse_m: 0.007494029801436203
  val_p95_m: 0.007468573105323891
  val_maximum_m: 0.10955820032334314
  resources:
    elapsed_seconds: 42.75992155598942
    peak_process_rss_bytes: 1017815040
    minimum_available_host_bytes: 12746555392
    gpu_jobs: 0
artifacts:
  run_dir: knowledge/runs/run-i936-condition-readout-r11-s936
  output_dir: /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/outputs/c936-r11-readout
  predictions: knowledge/runs/run-i936-condition-readout-r11-s936/output
parents:
- run-i936-h-dev-long-flow-regression-r10-s936-20260930
relations: []
papers: []
tags: []
date: '2026-09-30'
repro:
  commit: cf406ab46afb332ed905ca25b24b4c0a1d3b1c88
  branch: campaign930/i936-2-synthetic-diffusion
  command: timeout --signal=TERM --kill-after=10s 300s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.probe_conditioning_3d --dataset /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936
    --training-output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r10-s936-20k
    --output /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/outputs/c936-r11-readout
---

[事前固定計画](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5908211256)に従い、
20k flowの最終checkpointを固定してCPUで**1回だけ**実施した。GPU jobは0件。
[実測manifest](../../runs/run-i936-condition-readout-r11-s936/output/manifest.json)と
[再現情報](../../runs/run-i936-condition-readout-r11-s936/run.json)、head係数/特異値、全train/val予測を保存。

## 固定したテスト

本体encoderの全重みを凍結し、全125成分の非線形符号化・重み付きpool直後の128次元tokenから、
**その入力自身の全混合平均**を読む線形head＋切片（387係数）だけをfitした。
float64 SVD最小二乗、driver=gelsd、rcond=1e-12、ridgeなし。rank不足時の別solverへの切替なし。
64 train / 27,676 frameでfitし、同じ16 val / 6,383 frameの全frameを評価。
合成GT位置はfit教師に使わず、test NPZを開かない。checkpoint/seed/成分の選別なし。

実モデルの`encode_condition()`を共有しており、独立に似た符号化を再実装していない。
公開メソッドへの切り出し前後は20k重み・B2/T128/M64解析fixtureで位置/イベント出力がbit一致し、
state_dict keyも不変。学習本体・損失weight・入力bank・H生成既定を変更していない。

## 観測

| split / 可視camera数 | frame数 | 入力平均へのRMSE m | 誤差p95 m | 最大誤差 m |
|---|---:|---:|---:|---:|
| train / all | 27676 | 0.001649 | 0.003385 | 0.019250 |
| train / 0 | 1923 | 0.001676 | 0.003165 | 0.015782 |
| train / 1 | 746 | 0.003814 | 0.006871 | 0.014363 |
| train / 2 | 3704 | 0.003094 | 0.006452 | 0.015759 |
| train / 3 | 21303 | 0.001052 | 0.002076 | 0.019250 |
| val / all | 6383 | 0.007494 | 0.007469 | 0.109558 |
| val / 0 | 483 | 0.004568 | 0.010200 | 0.024545 |
| val / 1 | 467 | 0.026078 | 0.088348 | 0.109558 |
| val / 2 | 602 | 0.006306 | 0.014019 | 0.024414 |
| val / 3 | 4831 | 0.001192 | 0.002187 | 0.012040 |

事前基準の**val RMSE <= 0.10mを達成**した。最大誤差は0.1096mであり、全frameが0.10m以内とはしない。
train/valのRMSEは約1.65/7.49mm。3camera可視ではval約1.19mm、1cameraでは約26.08mm。
正規化round trip最大絶対誤差は3.55e-15m、raw特徴を全重みで集めて
平均を戻す最大絶対誤差は1.77e-06m。
SVD rankは129/129。係数と全特異値をhead.npzへ保存した。

これは**混合平均への復元誤差**であり、合成3D GTへの軌道精度ではない。
元の混合平均のGT RMSEは7.36mで、read-outを新しい3D refinerの性能改善とは主張しない。

## 解釈と次の因子

わずか387係数の固定headで、未fitのvalからもmm単位で入力平均を取り戻せた。
したがってpooled tokenへの大きな不可逆な平均情報の欠落や、この平均経路での単位取り違えは
今回のメートル単位の悪化を説明しない。これは最終flow encoder・旧bankのdevに限る結果で、
共分散の全情報、state/timeの符号化、Transformer以降、regression側encoderの正しさまでは証明しない。

[20kの曲線](000016-run-i936-h-dev-long-flow-regression-r10-s936-20260930.md)が示すtrain改善/val悪化と合わせ、
64 trainへの過学習を第一の説明として扱う。物理項はsmoothnessを改善したが、
正しい位置へ寄せる一般化は改善していない。物理項がその学習経路に及ぼす因果はまだ未検証。

次の候補は**既存checkpointのval推論文脈だけを全ラリーからT128へ変えるCPU対照**。
trainとvalの長さの差を、重みやlossを再学習せず切り分けられるため。
同じ16 valを全frame一度ずつ、末尾の重複は既定の中心距離・同点早い開始位置で固定し、
元の全ラリー予測と比較する。コード/費用の確定は次directiveで行い、今回追加実行していない。
それでも未達なら、同じ旧dev/初期化/窓順/2k更新でphysics weightだけ1e-4→1e-5に下げる比較を提案する。
x0-onlyと同時に変えたり、データ増量を併用したりしない。GPUの追加投入はしていない。

## 資源・入力交換

実測42.760秒、peak process RSS 1,017,815,040 bytes、
実行中の最低空きRAM 12,746,555,392 bytes。
CPU1 process/thread、CUDA無効、出力1,612,095 bytes。
全input/weight/source NPZの前後hashを確認し、予測・head・manifestをknowledgeへpromoteした。
fit後にvalで閾値や正則化を調整していない。

#935のユーザー判断Dに備え、既存`--calibration-report`がreport＋同階層bankのhash/schema/Kを固定することを確認。
新bankの公開hashはこの診断の開始時点で未投稿なので再生成0件。
公開後はbankを自分のcheckoutに固定して同じ96ラリーplan・H・seedで別出力へ生成し、
旧#959 devを対照として保持する。640件は最終#935出力待ち。
中央値の約13倍改善を全成分sigmaの一律縮小へ置換しない。
