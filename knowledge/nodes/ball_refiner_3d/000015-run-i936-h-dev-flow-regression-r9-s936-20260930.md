---
id: run-i936-h-dev-flow-regression-r9-s936-20260930
type: run
task: ball_refiner_3d
sequence: 15
recorded_at: '2026-09-30'
title: H dev固定2,000更新のflow/回帰：結果回収と単純ベースライン比較
issue: 936
provider: codex
session: 01a0f04f-8f4e-7823-ace0-31631c7a8788
date: '2026-09-30'
status: done
config:
  seed: 936
  updates: 2000
  batch_size: 8
  frames: 128
  stride: 128
  learning_rate: 0.0001
  weight_decay: 0.01
  evaluate_every: 500
  samples: 4
  steps: 8
  maximum_seconds: 3300
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
    elapsed_seconds: 335.4020270690089
    peak_allocated_bytes: 424876032
    peak_reserved_bytes: 450887680
    peak_device_used_bytes: 1751711744
    minimum_available_host_bytes: 20561420288
    driver_measurement: total-minus-free sampled each update and val rally; includes
      other usage; not a continuous driver peak
  parameters: 817285
  flow:
    updates_per_second: 12.942318959250878
    rmse_m_overall:
      count: 6383
      value: 8.011800133136243
    rmse_m_gap:
      count: 480
      value: 5.33229846687373
    rmse_m_no_evidence:
      count: 483
      value: 5.435105558650862
    rmse_m_event_pm5:
      count: 1103
      value: 8.156155651396896
    acceleration_all:
      count: 6351
      mean: 1177.3399304909617
      p50: 892.4546540963394
      p95: 2939.836601411649
    acceleration_free_flight:
      count: 3661
      mean: 1108.5984107362299
      p50: 853.8037510232515
      p95: 2767.008767823539
    jerk_all:
      count: 6335
      mean: 124601.95197074056
      p50: 96496.41124453161
      p95: 306963.58150165377
    jerk_free_flight:
      count: 3534
      mean: 117480.21432022352
      p50: 92693.3750201657
      p95: 287960.29589175613
    reprojection_px_all:
      count: 17528
      mean: 150.1661057827094
      p50: 110.38947948091746
      p95: 420.76845059477216
    behind_all:
      count: 17528
      invalid_count: 0
      fraction: 0.0
    reprojection_px_observed:
      count: 16164
      mean: 151.32370308017778
      p50: 111.05659537526202
      p95: 429.0990955568348
    behind_observed:
      count: 16164
      invalid_count: 0
      fraction: 0.0
    reprojection_px_gap:
      count: 1364
      mean: 136.44806860068704
      p50: 105.01123058599225
      p95: 352.9861229952513
    behind_gap:
      count: 1364
      invalid_count: 0
      fraction: 0.0
    reprojection_all_defined: true
  regression:
    updates_per_second: 13.251838293713607
    rmse_m_overall:
      count: 6383
      value: 8.20495944738825
    rmse_m_gap:
      count: 480
      value: 5.220355461775475
    rmse_m_no_evidence:
      count: 483
      value: 5.296787030820195
    rmse_m_event_pm5:
      count: 1103
      value: 8.345258640194045
    acceleration_all:
      count: 6351
      mean: 1734.0031827338967
      p50: 1406.5135525664402
      p95: 4057.249234941264
    acceleration_free_flight:
      count: 3661
      mean: 1585.7882228275141
      p50: 1314.0055499763318
      p95: 3640.058369617697
    jerk_all:
      count: 6335
      mean: 183537.17491639874
      p50: 152594.78103233152
      p95: 429873.5876960496
    jerk_free_flight:
      count: 3534
      mean: 167385.66516142897
      p50: 141804.6101430694
      p95: 376060.8623324119
    reprojection_px_all:
      count: 17525
      mean: 150.61123034492675
      p50: 109.05585061422282
      p95: 431.1602829901615
    behind_all:
      count: 17528
      invalid_count: 3
      fraction: 0.00017115472387037883
    reprojection_px_observed:
      count: 16164
      mean: 150.35989486342743
      p50: 110.07034816543565
      p95: 431.0497051736248
    behind_observed:
      count: 16164
      invalid_count: 0
      fraction: 0.0
    reprojection_px_gap:
      count: 1361
      mean: 153.5962316108743
      p50: 99.35869340357235
      p95: 436.0993260172029
    behind_gap:
      count: 1364
      invalid_count: 3
      fraction: 0.0021994134897360706
    reprojection_all_defined: false
repro:
  commit: a0ef7dd67f72097fff50afc8018faff93078a3e7
  branch: campaign930/i936-2-synthetic-diffusion
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 3540s env CUDA_VISIBLE_DEVICES=0
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.training_dev_3d --dataset /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936
    --config /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/src/tasks/ball_refiner/refiner_3d/training_dev.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r9-s936
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-i936-h-dev-flow-regression-r9-s936-20260930
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790742870654274948_3515634_i936-h-dev-flow-regression-r9-s936-20260930.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r9-s936
  predictions: knowledge/runs/run-i936-h-dev-flow-regression-r9-s936-20260930/output
  curves: knowledge/runs/run-i936-h-dev-flow-regression-r9-s936-20260930/output
parents:
- run-i936-h-dev-r9-s936
relations: []
papers: []
tags: []
---

固定64 train / 16 val、全125成分、seed936、fp32、B8/T128、817,285 parameters。
[回収記録](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/collection.json)に
queue done、全出力のSHA/bytes、checkpoint以外の保存先、両armの同一初期重み・
全2,000更新の同一window順の照合を記録した。testを開いた記録はない。
実行時のrepro bundleはcommit a0ef7dd6のまま保存した。

[manifest](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/output/manifest.json)の
所要時間は335.40秒、flow/regressionの更新速度は12.94/13.25 updates/s。
peak allocated 424,876,032 bytes、reserved 450,887,680 bytes、driver標本最大
1,751,711,744 bytes。driverは他用途を含む標本値で瞬間最大の保証ではない。
最小空きRAM20,561,420,288 bytes、出力17,308,593 bytes（manifest内の計測時点）。

16 val / 6,383 frameの2k時点RMSEはflow 8.012m、regression 8.205m。
初期14.05/16.47mから下がるが、flowの加速度p95は2,939.8m/s²、jerk p95は
306,963.6m/s³で、精度・物理的妥当性の合格ではない。回帰は3件のbehind-camera予測があり、
その再投影統計は正depth条件付きである。

[flow曲線](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/output/flow/curves.png)と
[回帰曲線](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/output/regression/curves.png)、
全update JSONL、全評価時点manifest、最終val予測を保存した。TensorBoardは使用していない。
flowの1k→2k RMSEは7.790→8.012m、回帰は7.253→8.205mで単調改善していない。
総lossの低下だけから、入力分布を読む以上の学習やパレート優位を主張しない。

run 10では同じ16 valで、3D混合平均・最大重み成分平均・#929重力RTSを無学習で採点する。
次に更新数だけを延長し、同じ設定で20kまで学習する。最終#935較正、640ラリー、実Meiji、
pipeline統合は対象外。較正・seed・成分・損失を変更しない。
