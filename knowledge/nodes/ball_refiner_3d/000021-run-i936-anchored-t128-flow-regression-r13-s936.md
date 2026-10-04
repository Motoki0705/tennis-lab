---
id: run-i936-anchored-t128-flow-regression-r13-s936
type: run
task: ball_refiner_3d
sequence: 21
recorded_at: '2026-09-30'
title: T128の20k比較はRMSE改善・RTSに対する粗さと再投影の優位未達
issue: 936
provider: codex
status: done
date: '2026-09-30'
session: 01a0f22b-2c7e-7f40-b507-54d94792695f
config:
  source: src/tasks/ball_refiner/refiner_3d/training_dev_anchored_t128.yaml
  primary_update: 20000
  train_frames: 128
  validation_frames: 128
metrics:
  flow_rmse_m: 3.651
  regression_rmse_m: 2.651
  wall_seconds: 3791.9971940349787
  peak_device_used_bytes_sampled: 1772683264
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


## Run 14での回収・事前規則の適用

2026-10-01にqueueのdone原本・clean repro（実行commit `98e67e957d1cd039f0734697f78210640d24510e`）を回収した。
[collection.json](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/collected/collection.json)は
全成果物・両checkpoint/初期重みのhash、資源、15評価軸ごとの数値と合否を保持する。
[collect.py](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/collect.py)で全16val×5更新×2armの160保存予測を再計算し、
GT/mask/camera/分母・平均と全sampleの全体/層別指標・出力Markdownが一致した。
run12の混合平均/RTS/GTも同じvalから再計算して一致した。test配列は0件。

[全評価時点・全層の表](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/collected/comparison.md)と
[数値原本](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/collected/comparison.json)が結果の正本。
主判定は20kで固定し、flow平均・全sample・回帰の**すべてが事前の優位条件を満たさない**。
flow/回帰のRMSEは3.651/2.651mで混合平均4.994/RTS4.408mより低い。
しかし自由飛行加速度p95は1,847.887/2,203.622対RTS158.108m/s²（11.7/13.9倍）、
jerk p95は199,756/241,101対7,133m/s³（28.0/33.8倍）。
再投影mean/p50は両armでRTSより悪く、behindはflow平均6/17,528、全sample24/70,112、回帰1/17,528でゼロ条件に失敗した。
flowのcamera1 RMSEは12.667m、camera3は0.880m。良い多数camera層でも粗さが残る。
flow平均と全sampleの指標は近いが、sample選択による改善とは扱わない。

10k→20kのflow RMSEは3.106→3.651mに悪化、回帰は2.643→2.651mで停滞。
中間checkpointは診断であり主判定を10kへ差し替えない。diffusion固有の優位も未達。
合成dev上での失敗を記録し、Hの選定・#959 control・最終test・Meiji/pipelineの判断は変更しない。

実測は3,791.997秒（63.20分）、allocated422,902,272 bytes、reserved471,859,200 bytes、
GPU driver使用量の最大標本値1,772,683,264 bytes、最小空きRAM18,144,673,792 bytes。
更新速度flow10.765/回帰11.128 updates/s、20k検証は9.940/1.139秒（全6,383frame、指標と保存を含む）。
出力実サイズ45,535,686 bytes。driver値は各update/valでの標本で連続peakではない。
TensorBoardはなく、JSONLと[flow曲線](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/collected/flow/curves.png)・
[回帰曲線](../../runs/run-i936-anchored-t128-flow-regression-r13-s936/collected/regression/curves.png)を保持する。
次は保存済み20kをCPUで継ぎ目/窓内・camera数・sample平均との差に分け、損失の重み付き実測から追加一因子試験を決める。
