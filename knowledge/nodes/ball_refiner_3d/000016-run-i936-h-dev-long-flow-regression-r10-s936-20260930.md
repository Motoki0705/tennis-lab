---
id: run-i936-h-dev-long-flow-regression-r10-s936-20260930
type: run
task: ball_refiner_3d
sequence: 16
recorded_at: '2026-09-30'
title: 20k更新でtrainは改善・valは悪化：全曲線と予測の回収
issue: 936
provider: codex
session: 01a0f09e-033b-7153-90ed-4edba72255e4
date: '2026-09-30'
status: done
config:
  seed: 936
  updates: 20000
  batch_size: 8
  frames: 128
  stride: 128
  learning_rate: 0.0001
  weight_decay: 0.01
  evaluate_updates:
  - 0
  - 2000
  - 5000
  - 10000
  - 15000
  - 20000
  samples: 4
  steps: 8
  maximum_seconds: 5100
  allocator_limit_gib: 6.0
  maximum_device_bytes: 10000000000
  expected_counts:
    train: 64
    val: 16
    test: 16
  model:
    width: 128
    layers: 4
    heads: 4
    feedforward_multiplier: 4
    time_frequencies: 8
    dropout: 0.0
  loss:
    x0: 1.0
    reprojection: 0.01
    physics: 0.0001
    event: 0.1
metrics:
  resources:
    elapsed_seconds: 3441.221497874998
    peak_allocated_bytes: 424876032
    peak_reserved_bytes: 450887680
    peak_device_used_bytes: 1751711744
    minimum_available_host_bytes: 17750740992
    driver_measurement: total-minus-free sampled each update and val rally; includes
      other usage; not a continuous driver peak
  arms:
    flow:
      updates_per_second: 11.969772702933147
      final_val_rmse_m: 9.795937909880458
    regression:
      updates_per_second: 11.906403839575681
      final_val_rmse_m: 9.152596656370205
repro:
  commit: 91193a990318cadc756e237cb00fc51aab7a87e3
  branch: campaign930/i936-2-synthetic-diffusion
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 5340s env CUDA_VISIBLE_DEVICES=0
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.training_dev_3d --dataset /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936
    --config /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/src/tasks/ball_refiner/refiner_3d/training_dev_long.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r10-s936-20k
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-i936-h-dev-long-flow-regression-r10-s936-20260930
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790745177452488394_4095209_i936-h-dev-long-flow-regression-r10-s936-20260930.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r10-s936-20k
  curves: knowledge/runs/run-i936-h-dev-long-flow-regression-r10-s936-20260930/train-val.png
  predictions: knowledge/runs/run-i936-h-dev-long-flow-regression-r10-s936-20260930/output
parents:
- run-i936-h-dev-flow-regression-r9-s936-20260930
relations: []
papers: []
tags: []
---

各arm20,000更新、64 train / 16 val、seed936、Hの全125成分、暫定#959 bank。
変更は更新数・評価日程・実行予算のみ。[回収台帳](../../runs/run-i936-h-dev-long-flow-regression-r10-s936-20260930/collection.json)に
queue done、complete、同一初期化、全20kの同一窓順、全出力SHA/bytesを保存した。
旧runとの最初の2k train lossと0/2k val全指標は厳密一致。
全6評価時点・両armの保存予測から全体/可視camera層別・mean/sample/truth全指標を再計算し厳密一致した。
実行時91193a99のreproは改変せず保存。checkpointは元出力に保持しhashだけを登録、test読込0。

## Trainとvalidationの推移

[同じ図のtrain/val曲線](../../runs/run-i936-h-dev-long-flow-regression-r10-s936-20260930/train-val.png)と
[全損失の数値CSV](../../runs/run-i936-h-dev-long-flow-regression-r10-s936-20260930/train-val.csv)を保存。
元の全update JSONL・PNGも各armのoutputに保持し、TensorBoardは未使用。
trainは直前500 updateのonline pre-update lossを実frame数で加重したもの。
valは全ラリーを固定checkpointで測り、flowは固定noisy x_t/timeでのx0 lossである。
trainはT128窓・変化する重み/noise、valは最大T512全ラリーであり、比を純粋な汎化gapと読まない。
val生成RMSEは別に固定noiseの4 sample×8 stepで計測する。

| arm | updates | train x0 | val x0 | train physics | val physics | val RMSE m |
|---|---:|---:|---:|---:|---:|---:|
| flow | 2000 | 0.017907 | 0.118240 | 66.093 | 73.634 | 8.012 |
| flow | 5000 | 0.007787 | 0.137699 | 21.416 | 37.043 | 8.374 |
| flow | 10000 | 0.004168 | 0.179973 | 11.637 | 25.555 | 9.188 |
| flow | 15000 | 0.002480 | 0.205421 | 8.412 | 21.957 | 9.716 |
| flow | 20000 | 0.001753 | 0.219647 | 6.799 | 18.528 | 9.796 |
| regression | 2000 | 0.016707 | 0.158867 | 51.986 | 79.313 | 8.205 |
| regression | 5000 | 0.006148 | 0.200946 | 16.856 | 45.281 | 9.228 |
| regression | 10000 | 0.002761 | 0.204686 | 9.650 | 35.361 | 9.313 |
| regression | 15000 | 0.001721 | 0.203675 | 7.461 | 32.814 | 9.290 |
| regression | 20000 | 0.001402 | 0.197683 | 6.450 | 30.184 | 9.153 |

## 観測と因果の限界

両armともtrainのx0/総lossは低下する一方、2k→20kのvalは悪化した。
flowのtrain x0は約10倍低下しval x0は約1.86倍になるため、過学習を支持する。
trainの精度も悪化している単純な「物理項が全軌道を間違った形へ押している」説明には合わない。
ただし物理項が汎化や学習経路を害する可能性は残り、同時推移から因果は分離できない。
T128/全ラリーの文脈差も未分離で、純粋な同条件train/val評価ではない。

valのflow再投影lossは13.53→13.60とほぼ停滞し、physicsは73.63→18.53に低下。
physicsの重み付き値は0.00736→0.00185で、x0増分0.1014を総loss上で打ち消していない。
項の値とgradientの大きさ/方向は異なるので、物理weightの強弱をこの数字だけで決めない。
回帰は5k以降ほぼ横ばいだが2kより悪く、flowのsample手順だけを唯一の原因ともできない。
どの保存時点も混合平均7.363m・RTS6.972mを下回らず、更新延長だけの採用根拠はない。

## 20kの精度と滑らかさ

| 最終arm | 全体RMSE m | gap RMSE m | event±5 RMSE m | 加速度p95 m/s²（全体/自由飛行） | 再投影p95 px |
|---|---:|---:|---:|---:|---:|
| flow | 9.796 | 4.566 | 9.774 | 1255.0 / 881.7 | 524.32 |
| regression | 9.153 | 4.510 | 9.432 | 2061.1 / 1518.1 | 516.08 |

両armともbehind-camera 3/17,528件を保持しているため再投影は正depth条件付き。
flowの加速度は2k 2939.8から低下しRTS 2020.9も下回るが、RMSE/再投影で劣る。
したがって「20kでもRTSが全軸で支配する」とは言わない。真値の自由飛行加速度p95=12.97からは大きく離れる。

| 可視camera数 | frame数 | 混合平均RMSE m | flow 20k RMSE m | 回帰20k RMSE m |
|---:|---:|---:|---:|---:|
| 0 | 483 | 5.145 | 4.688 | 4.639 |
| 1 | 467 | 18.232 | 19.017 | 17.429 |
| 2 | 602 | 12.911 | 13.339 | 11.920 |
| 3 | 4831 | 4.009 | 8.214 | 7.840 |

3cameraの多数frameでの悪化が残り、flowの1cameraでの2k時点の改善も失われた。
mask由来の可視camera数であり、amodal存在確率とは区別する。

## 資源と次のテスト

実測3,441.22秒（57.35分）、flow/regression=11.97/11.91 updates/s。
peak allocated/reserved=424,876,032/450,887,680 bytes、driver標本最大1,751,711,744 bytes。
最低空きRAM17,750,740,992 bytes。元出力46,130,628 bytes、checkpoint以外のpromote32,962,026 bytes。
driver値は連続監視のpeakではない。manifestのoutput_bytesは最終書込前なので実測台帳との差392 bytesを保持した。

次は[事前固定したCPU read-out](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5908211256)。
最終flow条件encoderを凍結し、全trainのtokenから全混合平均を線形に読み出し同じvalで測る。
loss weight変更やデータ追加より先に、後続実験が依存するencodingを安価に検査する。
成功なら大きな平均情報の欠落は支持されず、失敗だけで非線形read-outの不可能性は断定しない。
このrunの回収は新たな学習/因子変更ではない。640件は最終#935出力待ち、test/実Meiji/pipelineは未実施。
[D採用](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5908081470)に従い新bank確定後にdevを別directoryへ再生成し、今回の旧入力を対照として保持する。
