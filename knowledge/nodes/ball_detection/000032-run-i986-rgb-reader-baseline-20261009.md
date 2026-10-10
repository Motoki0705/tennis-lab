---
id: run-i986-rgb-reader-baseline-20261009
type: run
task: ball_detection
sequence: 32
recorded_at: '2026-10-09'
title: native RGB readerのCPU再計測：hash・RGB配置・IPC
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  workers: 8
  windows: 96
  repeats: 2
  device: cpu
metrics:
  first_windows_per_second: 1.0575643884335877
  replay_windows_per_second: 4.678582627679386
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 timeout 600 .venv/bin/python
    tests/benchmarks/ball_mdd_cpu_profile.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-baseline-v1/normal.json
    --case normal --workers 8 --windows 96
artifacts:
  run_dir: knowledge/runs/run-i986-rgb-reader-baseline-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791503084904813206_1090616_i986-native-reader-baseline-v1.log
parents:
- run-i986-native-rgb-compile-20261009
relations: []
papers: []
tags:
- input-performance
- cpu
- diagnostic
---

モデル内MDDへ移した後も、native RGB本学習は850更新の時点で平均1.460窓/秒となり、ユーザー指示で入力経路の改善へ戻った。checkpointは未保存。

CPUだけで同じ96窓を8 workers・prefetch 1で2回読み、初回は1.058窓/秒、同じworkerでの再読込は4.679窓/秒だった。初回のworker内平均CPU時間はhash 1.924秒、JPEG decode等0.124秒、残る入力準備約0.921秒、collate 1.199秒。並列workerのservice timeなので、これらを足してmainのstep時間とはみなさない。

MDD生成を移しただけでは、clip全体hashの重複とRGB CHW配置・大きなbatchのコピーが残る。次は検証成功の共有、shared-memory collate、圧縮JPEGを渡す方式を試す。

[全計測](../../runs/run-i986-rgb-reader-baseline-20261009/normal.json)にrow別CPU/壁時間を保存。GPU・pin memory・H2Dは測っていない。再読込も初回hash時間を除外しており、総時間がその速度になるという意味ではない。TensorBoardなし。reproは、元processが継承したPYTHONPATHを明示した修復版と原文を併記した。
