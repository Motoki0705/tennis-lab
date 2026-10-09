---
id: run-i986-nvjpeg-long-reader-20261009
type: run
task: ball_detection
sequence: 39
recorded_at: '2026-10-09'
title: 1,020窓でreader待ち再発：短い比較だけでは不十分
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  model: conv2d-query_only
  precision: bf16
  jpeg_decoder: nvjpeg
  image_prefetch: true
  batch_size: 1
  workers: 2
metrics:
  windows_per_second: 5.649940255363217
  mean_reader_wait_seconds: 0.12040680539661162
  preverification_seconds: 605.7640301250067
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 2400 .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-sustained-v1/report.json
    --mode pipeline --workers 2 --prefetch-factor 2 --pin-memory --precision bf16 --compile-mode default --jpeg-decoder
    nvjpeg --image-prefetch --preverify --batch-size 1 --warmup 12 --steps 1008
artifacts:
  run_dir: knowledge/runs/run-i986-nvjpeg-long-reader-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791506286423702361_1295939_i986-nvjpeg-sustained-v1.log
parents:
- run-i986-jpeg-prefetch-sweep-20261009
relations: []
papers: []
tags:
- input-performance
- nvjpeg
- reader-diagnostic
---

同じnvJPEG＋先読み・BS1・2 workersを1,020窓へ広げると、12 warmup後の1,008更新は5.650窓/秒へ落ちた。CPU reader待ちは平均120.4ms、stepは177.0msであり、短い204窓だけでCPUボトルネック解消とは判断できない。本学習を再開せず調査を継続した。

501 clipの事前検証に605.76秒を要した。最長のreader待ちは1.70秒で、100更新ごとの速度も約4.36〜8.43窓/秒へ変動した。入力準備完了待ちと、producer内で測るCPU reader待ちは重なりがあるため加算しない。

[全step](../../runs/run-i986-nvjpeg-long-reader-20261009/report.json)。精度試験ではなく、実行自体は完了したが性能の採用基準は満たさなかった。TensorBoardなし。原scriptと同一YAMLの参照修復を保存した。
