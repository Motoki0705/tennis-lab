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
  baselines:
    mixture_mean:
      rmse_m_overall:
        count: 6383
        value: 7.3627655769321105
      rmse_m_gap:
        count: 480
        value: 4.930544540536722
      rmse_m_no_evidence:
        count: 483
        value: 5.14498915031589
      rmse_m_event_pm5:
        count: 1103
        value: 6.8948798301230365
      acceleration_all:
        count: 6351
        mean: 6974.49131124366
        p50: 3443.042385922998
        p95: 25943.211606806155
      acceleration_free_flight:
        count: 3661
        mean: 6970.03533054928
        p50: 3530.3890146789345
        p95: 26127.440598763944
      jerk_all:
        count: 6335
        mean: 750762.8648441226
        p50: 400336.011798243
        p95: 2650194.7706126478
      jerk_free_flight:
        count: 3534
        mean: 749532.0979248089
        p50: 403158.03065363213
        p95: 2635681.5604524114
      reprojection_px_all:
        count: 17528
        mean: 64.74076206169173
        p50: 30.25915704572343
        p95: 239.59019716583566
      behind_all:
        count: 17528
        invalid_count: 0
        fraction: 0.0
      reprojection_px_observed:
        count: 16164
        mean: 63.040078298626824
        p50: 27.742172637631146
        p95: 240.77381351840694
      behind_observed:
        count: 16164
        invalid_count: 0
        fraction: 0.0
      reprojection_px_gap:
        count: 1364
        mean: 84.89461275537309
        p50: 64.57866770803619
        p95: 218.71540501983768
      behind_gap:
        count: 1364
        invalid_count: 0
        fraction: 0.0
      reprojection_all_defined: true
    top_component:
      rmse_m_overall:
        count: 6383
        value: 7.352694182285894
      rmse_m_gap:
        count: 480
        value: 5.033377481722692
      rmse_m_no_evidence:
        count: 483
        value: 5.233741938258391
      rmse_m_event_pm5:
        count: 1103
        value: 6.684102553733526
      acceleration_all:
        count: 6351
        mean: 9005.103343158895
        p50: 3805.482530565334
        p95: 34648.36242492706
      acceleration_free_flight:
        count: 3661
        mean: 8915.228840207477
        p50: 3889.2211108567735
        p95: 34662.826954217584
      jerk_all:
        count: 6335
        mean: 970408.167587329
        p50: 456156.03591909
        p95: 3609783.2002888294
      jerk_free_flight:
        count: 3534
        mean: 957976.8797390638
        p50: 451435.7871483522
        p95: 3543936.5327060027
      reprojection_px_all:
        count: 17528
        mean: 65.52869939699083
        p50: 26.183887538117844
        p95: 258.51146832416816
      behind_all:
        count: 17528
        invalid_count: 0
        fraction: 0.0
      reprojection_px_observed:
        count: 16164
        mean: 63.627839954325324
        p50: 23.861037856334303
        p95: 258.15351467330595
      behind_observed:
        count: 16164
        invalid_count: 0
        fraction: 0.0
      reprojection_px_gap:
        count: 1364
        mean: 88.05471994775709
        p50: 60.1283684422108
        p95: 260.00893281021973
      behind_gap:
        count: 1364
        invalid_count: 0
        fraction: 0.0
      reprojection_all_defined: true
    mixture_mean_rts:
      rmse_m_overall:
        count: 6383
        value: 6.972419048373081
      rmse_m_gap:
        count: 480
        value: 4.564751888436972
      rmse_m_no_evidence:
        count: 483
        value: 4.817731088897052
      rmse_m_event_pm5:
        count: 1103
        value: 6.149617574410649
      acceleration_all:
        count: 6351
        mean: 338.7002870368689
        p50: 35.233599514392175
        p95: 2020.8756429884097
      acceleration_free_flight:
        count: 3661
        mean: 300.5004071316494
        p50: 35.361734885536954
        p95: 1759.738053828293
      jerk_all:
        count: 6335
        mean: 34558.64052550295
        p50: 307.3369533919955
        p95: 241286.89215266568
      jerk_free_flight:
        count: 3534
        mean: 30811.898630129086
        p50: 291.73678399580103
        p95: 222786.23134554582
      reprojection_px_all:
        count: 17528
        mean: 59.920999195003496
        p50: 34.052844576844535
        p95: 202.86218465645723
      behind_all:
        count: 17528
        invalid_count: 0
        fraction: 0.0
      reprojection_px_observed:
        count: 16164
        mean: 58.8988771283387
        p50: 32.012617626437205
        p95: 203.58005111882983
      behind_observed:
        count: 16164
        invalid_count: 0
        fraction: 0.0
      reprojection_px_gap:
        count: 1364
        mean: 72.03359529879357
        p50: 54.237767817555365
        p95: 190.70193492360008
      behind_gap:
        count: 1364
        invalid_count: 0
        fraction: 0.0
      reprojection_all_defined: true
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

[同一16 valの比較表](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/baselines/comparison.md)と
[全指標/層別/出自](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/baselines/comparison.json)を
run 10で追加。元dataset manifest/全val SHAと予測内のGT・camera・maskを照合し、
保存したflow/回帰のmean/sample全指標が旧manifestと一致した。CPU baselineは1.09秒（reader検証を除く）。

3D混合平均7.363m、最大重み成分7.353m、混合平均の#929 RTS6.972mに対し、
2k flow8.012m/回帰8.205mは全体RMSEで劣る。3camera可視4,831frame（75.69%）では
混合平均4.009mに対しflow6.400m/回帰6.776mへ悪化する。一方1camera可視467frameでは
18.232mから14.394/15.144mへ改善する。『何も学んでいない』とも『入力以上に有用』とも一括りにしない。
RTSは全体RMSE/再投影p95/加速度p95で両学習armを上回るが、それでも加速度p95は
2,021m/s²（free1,760）と真値14.48（free12.97）から大きく離れている。
この暫定入力・16ラリー・1seedでの結果であり、本番のパレート優位の証拠ではない。

RTSの5関数は#929 commit 0f124818のASTと完全一致（[照合](../../runs/run-i936-h-dev-flow-regression-r9-s936-20260930/baselines/rts-provenance.json)）。
既定のposition sigma0.07m、acceleration sigma15m/s²、Huber0.12m、入力由来のイベントで無調整。
全frameの分布平均を支持し、教師イベントや可視性による除外を使わない。pipelineの幾何gateは
診断化し、bounds違反355frame、速度65m/s超259区間、検出イベント313frameを除外せず採点した。
多い偽イベントで端点が固定されるためRTSにもジッターが残り得るが、これは原因仮説である。
[暫定判断](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5904228981)を参照。

次は更新数だけ20kへ延長し、0/2k/5k/10k/15k/20kを同じ16 valで固定評価する。
seed/モデル/窓/optimizer/損失/較正/全成分は不変。評価日程とwall budgetは実験に合わせて延長する。
最終#935較正、640ラリー、実Meiji、pipeline統合は対象外。

20kで混合平均にも勝てない場合の次run候補（まだ実験していない）:
- 条件encoding: 重み付き非線形poolが平均/共分散情報を失う可能性。固定train prefixで
  condition→混合平均を小headで再構成し、identity精度に達するかCPU短時間で検査する。
- 正規化: meanと全covarianceの尺度、巨大共分散によるactivation支配の可能性。
  同じval入力で正規化round trip・各特徴の分位点/層別activationをCPUで監査する。教師でscaleを調整しない。
- 損失weight: 暫定較正の再投影/弱い重力priorがx0位置学習と競合する可能性。
  同一固定train batchで4項別のgradient norm/cosineを測る。次にx0-only短時間対照を1因子として申請する。
- 窓長: trainはT128窓、valは最大T512全ラリーで文脈長が異なる。
  保存checkpointを同じvalのT128窓・決め打ち中心採用でCPU評価し、全ラリー推論と比較する。
これらは20k結果を回収してから優先順位を決め、一度に複数因子を変更しない。
