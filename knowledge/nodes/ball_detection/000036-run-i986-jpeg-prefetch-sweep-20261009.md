---
id: run-i986-jpeg-prefetch-sweep-20261009
type: run
task: ball_detection
sequence: 36
recorded_at: '2026-10-09'
title: 同一204窓で入力先読み・worker・BSを比較
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  precision: bf16
  compile_mode: default
  windows: 204
  warmup_windows: 12
  seed: 42
  verification_workers: 8
metrics:
  async-w2-b2_windows_per_second: 13.950377826177764
  async-w2-b1_windows_per_second: 15.510012781648904
  async-w4-b1_windows_per_second: 14.750450539806693
  async-prepared-b1_windows_per_second: 14.731525719315702
  sync-w2-b1_windows_per_second: 10.114670252624327
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 1800 .venv/bin/python tests/benchmarks/ball_reader_sweep.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-sweep-v2
artifacts:
  run_dir: knowledge/runs/run-i986-jpeg-prefetch-sweep-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791504953305738781_1220241_i986-reader-sweep-v2.log
parents:
- run-i986-nvjpeg-prepared-20261009
relations: []
papers: []
tags:
- input-performance
- nvjpeg
- diagnostic
---

BS1・2 workers・prefetch factor 2・CUDAで1 batch先読みを採用候補とする。同じ先読みをするRAM内JPEG基準と同水準に到達し、この短い比較ではCPU readerによる速度差が見えなくなった。

| 条件 | 窓/秒 | 入力準備完了待ち（ms/update） |
|---|---:|---:|
| async-w2-b2 | 13.950 | 42.28 |
| async-w2-b1 | 15.510 | 10.28 |
| async-w4-b1 | 14.750 | 8.94 |
| async-prepared-b1 | 14.732 | 10.58 |
| sync-w2-b1 | 10.115 | 10.79 |

全caseで同じ204窓を使い、先頭12窓をwarmupから除いた。BS2は6 warmup＋96計測updatesで、BS1の12＋192と窓数を揃えた。各caseを同じseed・scratch・AdamWで始め、compile counterをcaseごとにresetした。モデル・FPS・GTは共通。

先読み時の待ちはCPU readerと復号の準備を含み、CPU待ちだけではない。RAM対照でも約10.6msの待ちがある。reader込みの方が約5%速かったことをreaderの因果効果とはみなさず、単発計測の変動を含む同水準と解釈する。BS2と4 workersはこの試行では改善せず、学習条件はBS1を維持する。

177 clip・22.33GBのdual SHA-256検証に308.13秒を要した。検証成功を同じDatasetから各workerへ共有し、この費用をcaseごとに重複させなかった。モデルの定常速度と、この準備費用を含む総時間を区別する。GPU側にはnvJPEGのhost処理・色変換・stackもあり、GPU常駐RGBだけの旧21.26窓/秒と同じ仕事量ではない。

[比較図](../../runs/run-i986-jpeg-prefetch-sweep-20261009/performance.png)／[印刷用PDF](../../runs/run-i986-jpeg-prefetch-sweep-20261009/performance.pdf)。bundle内の各case JSONに全stepと同じ窓列hashを保存した。精度・収束・全epoch安定性は未評価で、[1,020窓の追試](000039-run-i986-nvjpeg-long-reader-20261009.md)ではreader待ちが再発したため、この短い比較だけでは採用しない。TensorBoardなし。原scriptとYAML参照修復を保持した。
