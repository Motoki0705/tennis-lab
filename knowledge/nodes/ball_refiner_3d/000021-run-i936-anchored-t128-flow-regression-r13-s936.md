---
id: run-i936-anchored-t128-flow-regression-r13-s936
type: run
task: ball_refiner_3d
sequence: 21
recorded_at: '2026-09-30'
title: 新bank devのT128学習・検証を揃えた20k比較をqueueへ登録
issue: 936
provider: codex
status: planned
date: '2026-09-30'
session: 01a0f22b-2c7e-7f40-b507-54d94792695f
config:
  source: src/tasks/ball_refiner/refiner_3d/training_dev_anchored_t128.yaml
  primary_update: 20000
  train_frames: 128
  validation_frames: 128
metrics: {}
repro:
  commit: 0c98a9cf
  branch: campaign930/i936-2-synthetic-diffusion
artifacts:
  run_dir: knowledge/runs/run-i936-anchored-t128-flow-regression-r13-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r13-anchored-s936-t128-20k
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790771078312809159_1781052_i936-anchored-t128-flow-regression-r13-s936-20260930.log
parents:
- run-i936-anchored-dev-comparison-r12-s936
- run-i936-context-t128-r12-s936
- run-i936-h-dev-long-flow-regression-r10-s936-20260930
relations: []
papers: []
tags: []
---

[enqueue前の比較定義・判定規則](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5911005227)を先に投稿した。
[固定plan](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/plan.json)に実行コマンド、
入力・設定hash、80件のtrain/val identity、費用見積・上限を記録する。
planはenqueue前のsnapshotとして保持する。2026-09-30 21:24:38 JSTに
job **1790771078312809159_1781052_i936-anchored-t128-flow-regression-r13-s936-20260930** を1件だけ登録した。
[queue登録票](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/queue.json)と
[job原本](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/queued.job)にcwd、log/repro、
#935の2件→#964 feature→本jobという順序を保存した。既存worker PID3216003を使用し、
他job/workerは操作していない。登録時点ではqueuedで、結果は未観測。
TensorBoardは使わず、jobは全updateのJSONL/PNGと
全評価時点の予測・指標を出力する。quota休止後にqueueのdone/failedを回収し、statusと証拠を更新する。

新bankの64train/16valで各20kのflowと同backbone回帰を学習する。
run10の設定との差はvalidation文脈T128と指定評価日程だけで、入力bankの変更と合わせて事前宣言した。
seed936、T128/B8、初期化・窓shuffle、全loss/optimizer/architectureを維持する。
Hの全125成分と未評価flagを保持し、seed選別・成分削除・数値失敗のretryを行わない。
比較対象の新旧devについて保存NPZ192件のhashを確認したが、test配列は開いていない。

同じ16val/6,383frameを全手法で採点する。混合平均・無調整#929 RTS・GT・
全評価時点のflow平均/全sampleと回帰を同じcomparison.json/mdへ保存する。
T128の境界・短い末尾の採用規則はrun12診断と共有し、差分/lossも継ぎ目を含む元系列で計算する。
noiseは実行device上で全系列に一度だけ生成して切り出す。run12 CPUとGPUのbit一致は主張しない。

主判定は固定20k。途中0/2k/5k/10kから最良checkpointを選ばない。
RMSEだけの改善やbehindを含む条件付き再投影をパレート優位としない。
flowの平均/全sampleと回帰を分け、GTの自由飛行加速度、camera数別誤差、欠損/event、
再投影とbehindの分母を併記する。詳細な合否軸は上記の事前投稿が正本。
旧run10からはbankと評価文脈が変わるためbank単独の因果効果は分離できない。
Hの方式選定、test品質、独立Meiji LOCO、最終pipeline採用はこの実験の結論に含めない。

見積60〜85分・peak VRAM2〜3GB・disk150MBはrun10実測57.35分/1.752GBからの外挿。
resource=allを1件、最大5,355秒（89分15秒）、allocator6GiB/driver10GB、空きRAM6GiB下限。
640生成を4CPUで継続するためnative threadと他CPU作業を1に制限する。
通常検証99tests（pytest -n1）、ruff/mypy/pre-commit成功。GPUの数値・資源は未測定。
[検証記録](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/verification.json)は
21:25時点の640生成28件成功/失敗0、空きRAM14.92GB、生成途中出力64.23MBを含む。
640生成入力36hashは不変で、GPU出力はまだ作られていない。全job完了まで実行sourceを固定する。
