---
id: run-i986-query-bf16-sustained-20261008
type: run
task: ball_detection
sequence: 27
recorded_at: '2026-10-08'
title: BF16・BS1・worker8の252 update連続GPU検証
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-08'
status: done
config:
  model: conv2d-query_only
  precision: bf16
  batch_size: 1
  num_workers: 8
  warmup_windows: 12
  measured_windows: 240
metrics:
  windows_per_second: 1.3326888369484708
  peak_reserved_bytes: 2581594112
  mean_loader_wait_seconds: 0.5622398347213675
repro:
  commit: c8e6810616f3f31f3cff3dee0df7376d7c9a7a9b
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 20s 1200s .venv/bin/python
    tests/benchmarks/ball_mdd_query_gpu.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/conv2d-query-only-gpu/20261008-bf16-sustained-v1/bs1-workers8.json
    --precision bf16 --batch-size 1 --mode pipeline --workers 8 --pin-memory --warmup
    12 --steps 240
artifacts:
  run_dir: knowledge/runs/run-i986-query-bf16-sustained-20261008
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791459169173930426_613598_i986-query-bf16-sustained-v1.log
parents: [run-i986-query-bf16-confirm-20261008, run-i986-query-bf16-cli-20261008]
relations: []
papers: []
tags: [diagnostic, throughput, bf16]
---

推奨候補のBF16・BS=1・worker=8で、全train母集団から等FPSで選んだ252窓を連続学習した。
最初の12 updateを除く240 updateで1.3327窓/秒、42.65入力frame/秒。
全loss/gradient normがfiniteで、最大reserved VRAMは2.404GiBのままだった。
全stepの記録は[measurement.json](../../runs/run-i986-query-bf16-sustained-20261008/measurement.json)。

96窓比較の1.8505窓/秒より遅かった。選んだ窓列が異なり、clip初回hashやI/O待ちの長いstepが
存在するため、速度のばらつきを無視しない。平均step 0.750秒中、reader待ちは0.562秒。
この検証は「安定して全epochで1.85窓/秒出る」根拠ではない。
時間予算には今回の1.33〜1.85窓/秒を使い、validation通しの実測は別途必要とする。

初回の運用候補をBF16・BS=1・worker=8、prefetch 1、pin memory有効とする。
BS2/worker8のCUDA unknown errorは未解決の失敗として残す。
収束・汎化・最適学習率・必要update数は未測定で、本学習は開始していない。
具体的な提案は[学習レシピ](../../../src/tasks/ball_detection/training/CONV2D_QUERY_ONLY_RECIPE.md)を正本とする。
test予測・checkpoint・TensorBoard曲線はこの連続診断では生成していない。
