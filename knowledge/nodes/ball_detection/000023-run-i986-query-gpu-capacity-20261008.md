---
id: run-i986-query-gpu-capacity-20261008
type: run
task: ball_detection
sequence: 23
recorded_at: '2026-10-08'
title: Conv2d＋query-onlyのGPU計算速度・VRAM検証
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-08'
status: done
config:
  model: conv2d-query_only
  loss: observed-only SmoothL1 beta=0.01
  data: ball-mix-v2 frozen 20261008-fps124-v1
  input_shape: [2, 32, 720, 1280]
  seed: 42
  memory_fraction_limit: 0.9
metrics:
  fp32_bs1_windows_per_second: 7.177
  bf16_bs4_windows_per_second: 7.394
  parameters: 1126762
repro:
  commit: c8e6810616f3f31f3cff3dee0df7376d7c9a7a9b
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 20s 2400s bash tests/benchmarks/ball_mdd_query_gpu_sweep.sh
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/conv2d-query-only-gpu/20261008-sweep-v1
artifacts:
  run_dir: knowledge/runs/run-i986-query-gpu-capacity-20261008
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791456329552391447_371303_i986-conv2d-query-gpu-bs-sweep-v1.log
parents: []
relations: []
papers: []
tags: [diagnostic, throughput, mdd, query-only]
---

RTX 5060 Ti（16GB）、PyTorch 2.13.0+cu130で、trainの実MDD入力をGPUに置いたまま
順伝播・SmoothL1・逆伝播・gradient clip・AdamW updateを測定した。
各caseは別process、seed 42、4 update warmup後16 updateの平均。重みは各caseで初期化し、
同じbatchを反復する。読み込み速度や精度の比較ではない。

| BS | FP32 window/s | FP32 reserved GiB | BF16 window/s | BF16 reserved GiB |
|---:|---:|---:|---:|---:|
| 1 | 7.177 | 3.402 | 7.046 | 2.404 |
| 2 | 6.456 | 6.064 | 7.270 | 4.762 |
| 4 | 6.587 | 12.887 | 7.394 | 9.482 |
| 6 | 6.640 | 13.588 | 6.811 | 14.209 |
| 8 | OOM | — | OOM | — |

VRAMはPyTorch allocatorの最大reservedで、OS/表示やCUDA contextの総量ではない。
allocatorをGPU容量の90%に制限したため、OOMはその制限下での結果。
表示用に約1.2GiBも使われており、BS=6を安定した本学習設定としては採らない。
FP32はmatmul TF32 off、cuDNN TF32 on、cuDNN benchmark off。

成功した8 caseは全updateのloss/gradient normがfiniteで、重み更新も確認した。
BF16の最高速度はFP32 BS=1より約3%高いにとどまる。
短い同一batch反復なので収束、汎化、最適LR、必要update数の証拠にはしない。
末尾のFP32/BF16予測差は同じ未成熟な重みでの数値差であり、GT誤差ではない。
本学習BSはreader込みの追加比較で決める。

元JSONは[measurements](../../runs/run-i986-query-gpu-capacity-20261008/measurements)、
入力はmanifest SHA-256 `035d3ab96807ace8e25ebe3a5d603f09a7c514802942170fdd5b8ad2854c949c`。
checkpoint・test予測・TensorBoard曲線は生成していない。
