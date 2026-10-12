---
id: run-i986-query-gpu-reader-20261008
type: run
task: ball_detection
sequence: 24
recorded_at: '2026-10-08'
title: Query-onlyのJPEG・MDD reader込み速度の初期比較
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-08'
status: done
config:
  model: conv2d-query_only
  precision: fp32
  frame_steps: [1, 2, 4]
  reader_prefetch_factor: 1
metrics:
  bs1_workers0_windows_per_second: 0.5721
  bs1_workers4_windows_per_second: 1.5374
repro:
  commit: c8e6810616f3f31f3cff3dee0df7376d7c9a7a9b
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 20s 2400s bash tests/benchmarks/ball_mdd_query_gpu_sweep.sh
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/conv2d-query-only-gpu/20261008-pipeline-v1
    pipeline
artifacts:
  run_dir: knowledge/runs/run-i986-query-gpu-reader-20261008
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791456659354412883_398125_i986-conv2d-query-gpu-pipeline-v1.log
parents: [run-i986-query-gpu-capacity-20261008]
relations: []
papers: []
tags: [diagnostic, throughput, input-pipeline]
---

実際のtrain reader、FPS sampler、H2D転送、optimizer updateを含めて72窓ずつ計測した。
各caseは4 batchのwarmupを除外し、pin memory、workerごとのprefetch 1、persistent workersを使用した。
worker内はOpenCV/PyTorch各1 thread、メインはPyTorch 2 thread。

| BS | worker 0 | worker 2 | worker 4 |
|---:|---:|---:|---:|
| 1 | 0.5721 | 0.8042 | 1.5374 |
| 2 | 0.5213 | 0.7738 | 1.1565 |
| 4 | 0.5143 | 0.6304 | 0.8379 |

単位はwindow/s。GPU常駐時の約7 window/sより大幅に低く、CPU/I/O待ちが支配的。
BS=1・worker=4では取得待ちが平均0.487秒、全stepは0.650秒だった。
BS増加よりreader並列化を優先する根拠になる。

この初期比較はwarmupがbatch単位のため、samplerの総窓予算と窓列がBS間で異なる。
初回clip全体hash、JPEG内容、OS page cacheも速度に影響するので、BS間の小さな差は確定判断しない。
次は全caseで同じ108窓（12 warmup＋96計測）へ揃え、worker 6/8も比較する。
全epochの定常速度やモデルの汎化性能は未測定。9 caseとも数値はfiniteでupdateが成立した。

元JSONは[measurements](../../runs/run-i986-query-gpu-reader-20261008/measurements)。
test評価・checkpoint・TensorBoard曲線は生成していない。
