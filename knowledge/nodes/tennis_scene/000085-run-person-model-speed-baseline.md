---
id: run-person-model-speed-baseline
type: run
task: tennis_scene
sequence: 85
recorded_at: '2026-11-25'
title: Meiji 4camの現行人物モデルPT32・PT16・TRT16速度基準
issue: 2496
provider: codex
date: '2026-11-24'
status: done
summary_impact: updated
config:
  gpu: NVIDIA GeForce RTX 5060 Ti
  driver: '621.24'
  cuda: '13.0'
  pytorch: 2.13.0+cu130
  tensorrt: 10.14.1.48.post1
  pytorch_matmul_and_cudnn_fp32_precision: ieee
  input_id: meiji_4cam/person-model-speed-v1
  input_manifest_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
  warmup: 30
  measured_blocks: 3
  repetitions_per_block: 100
  detection_batches: [1, 2]
  pose_crop_batches: [1, 4, 8]
  body_sequence_batches: [1, 2, 4]
  body_sequence_frames: 16
metrics:
  fixed_images: 96
  fixed_crops: 384
  fixed_body_sequences: 24
  verified_input_files: 728
  measured_cases: 39
  parity_passed_cases: 24
  parity_failed_cases: 15
  timed_repetitions: 11700
  vitpose_no_flip_b1_engine_build_ms: 74352.295336
repro:
  commit: 0ee6f8dde06c078b6ab40324f5416b68a9eaefc0
  command: >-
    env PYTHONPATH="$PWD" TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    .venv/bin/python -B -u knowledge/runs/run-person-model-speed-baseline/run.py
    --output "$PERSON_RUN_OUTPUT" --trt-root "$PERSON_TRT_ROOT"
    --external-assets "$PERSON_EXTERNAL_ASSETS" suite "$PERSON_BASELINE_GROUP"
artifacts:
  run_dir: knowledge/runs/run-person-model-speed-baseline
  inputs: knowledge/runs/run-person-model-speed-baseline/inputs/manifest.json
  case_table: knowledge/runs/run-person-model-speed-baseline/cases.csv
  case_records: knowledge/runs/run-person-model-speed-baseline/cases.json
  queue_receipts: knowledge/runs/run-person-model-speed-baseline/queue-receipts.json
  output_dir: .worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1
parents:
- run-i2297-clipreid-tensorrt
- run-i2336-vision-batch-sweep
- run-i2363-trt-player-video
- run-meiji-multiview-player-evaluation
relations:
- to: run-i2304-dino-tensorrt
  rel: compares
- to: run-i2303-vit-tensorrt
  rel: compares
- to: run-i2355-gvhmr-pred-cam-clamp
  rel: compares
papers: []
tags: []
---

## 考察 / Findings

DINO、ViTPose-Hのflipあり・なし、DINO＋ViTPose-H、HMR2特徴＋GVHMRで、39条件それぞれの300反復を取得した。PT32を参照に24条件がparityを通過し、15条件は許容差を超えた。OOM・変換失敗・子プロセスの異常終了は0件だった。queue 4ジョブは完了し、39件の子プロセス終了値をすべて0と確認した。実行可否は個々の `status` を参照する。 [全batchの速度・VRAM・判定](../../runs/run-person-model-speed-baseline/cases.csv) · [条件別記録](../../runs/run-person-model-speed-baseline/cases.json)

固定入力 `meiji_4cam/person-model-speed-v1` に全画面96枚、PT32 DINOの検出と実観測から作った人物crop 384件、`body_view_selection` から固定した16サンプルの身体24系列を保存した。manifestと4 indexのSHA-256、および参照先728ファイルの内容を照合した。ViTPoseのbatch 1の計時は固定順の先頭300 cropで、6 clip中5 clipをカバーする。batch 4/8の計時と全ViTPose条件のparityは6 clipの384 cropを使う。候補と比較するときはbatchごとの入力順・反復数も揃える。 [固定入力manifest](../../runs/run-person-model-speed-baseline/inputs/manifest.json) · [各条件の計時対象](../../runs/run-person-model-speed-baseline/cases.json)

### Batch 1の結果

GPUはモデルforwardのCUDA event区間、wallは同期済みの前後処理と転送を含むstage。throughputの単位はDINOおよび2段で画像、ViTPoseで人物crop、HMR2＋GVHMRでperson-frame。VRAMはtorchのallocated・reserved peakとNVMLのデバイス全体のサンプル最大値で、単位はMiB。実行不能行のVRAMは試行中の観測値。 [計測と記録方式](../../runs/run-person-model-speed-baseline/README.md)

| モデル | 条件 | parity | GPU P50 / P95 ms | wall P50 / P95 ms | items/s | torch A / R / NVML MiB |
|---|---|---|---:|---:|---:|---:|
| DINO | PT32 | 通過 | 268.09 / 276.36 | 289.97 / 308.11 | 3.42 | 1780 / 2142 / 2709 |
| DINO | PT16 | 通過 | 147.80 / 160.64 | 167.68 / 180.48 | 5.90 | 2102 / 2366 / 2806 |
| DINO | TRT16 | 実行不能 | — | — | — | 12 / 14 / 1384 |
| ViTPose-H flipあり | PT32 | 通過 | 53.05 / 53.80 | 57.48 / 58.61 | 17.37 | 2550 / 2622 / 3060 |
| ViTPose-H flipあり | PT16 | 通過 | 20.44 / 29.68 | 24.50 / 34.73 | 38.81 | 3764 / 3950 / 4388 |
| ViTPose-H flipあり | TRT16 | 通過 | 9.33 / 9.66 | 12.89 / 14.00 | 76.27 | 2 / 2 / 1752 |
| ViTPose-H flipなし | PT32 | 通過 | 31.06 / 32.21 | 35.45 / 37.36 | 27.96 | 2539 / 2606 / 3044 |
| ViTPose-H flipなし | PT16 | 実行不能 | — | — | — | 3755 / 3944 / 4382 |
| ViTPose-H flipなし | TRT16 | 実行不能 | — | — | — | 1 / 2 / 1694 |
| DINO＋ViTPose-H | PT32 | 通過 | 561.54 / 852.77 | 590.33 / 890.03 | 1.54 | 4276 / 4690 / 5130 |
| DINO＋ViTPose-H | PT16 | 実行不能 | — | — | — | 5811 / 6442 / 6884 |
| DINO＋ViTPose-H | TRT16 | 実行不能 | — | — | — | 14 / 36 / 2692 |
| HMR2特徴＋GVHMR | PT32 | 通過 | 403.92 / 407.99 | 767.84 / 814.05 | 20.77 | 3013 / 3230 / 3670 |
| HMR2特徴＋GVHMR | PT16 | 通過 | 141.09 / 146.82 | 509.66 / 555.17 | 31.12 | 4338 / 4484 / 4926 |
| HMR2特徴＋GVHMR | TRT16 | 実行不能 | — | — | — | 50 / 54 / 1994 |

### 出力差による実行不能

- DINO TRT16のB=1はIoU P05が0.92980、score差P95が0.01335で、許容範囲（IoU 0.98以上、score差0.01以下）を外れた。DINO＋ViTPose-HのTRT16も同じ検出差を含み、PT16の2段合計はposeの最大差がB=1で30.64 px、B=2で39.43 pxに達した。2段合計では検出ごとに実際にcropを作るため、入力boxの変化を含めた出力差になる。 [条件別parity](../../runs/run-person-model-speed-baseline/cases.json)
- ViTPose-HのflipなしB=1はPT16・TRT16とも座標差P95が0.04 px未満だった。最大差はそれぞれ12.64 px・20.55 pxで、5 pxの基準を超えた。flipありの3 backendは全batchで適合した。 [条件別parity](../../runs/run-person-model-speed-baseline/cases.json)
- HMR2特徴＋GVHMRのTRT16 B=1は特徴cosine P05 0.999990、回転差P95 0.179°、カメラ座標の関節距離P95 21.79 mmだった。骨盤を原点に揃えた距離P95は1.05 mmで、カメラ座標の並進差P95は21.65 mmだった。10 mmの基準はカメラ座標で判定した。 [条件別parity](../../runs/run-person-model-speed-baseline/cases.json)

### 先行計測との照合

以前の同期込みforward中央値に合わせ、今回も `forward` のhost wall中央値を照合した。DINO PT32 B=1は #2304 の268.15 msと今回の268.15 msが小数第2位まで一致した。PT16は #2336 の150.43 msに対して147.86 ms。TRT16 GridSampleの診断値は #2304 の静的planでは56.98 ms、#2336 の動的planでは74.98 ms、今回の動的planでは75.12 msだった。同じ750×1333でも静的・動的profileと入力集合が異なる。TRT16のIoUとscoreの許容差超過は両先行runと同じ傾向だった。 [DINO単体](000078-run-i2304-dino-tensorrt.md) · [batch掃引](000080-run-i2336-vision-batch-sweep.md)

ViTPose-H flipありのPT32 B=1は #2303 の54.423 msに対して53.13 ms。TRT16は #2303 の8.468 msに対して9.38 msで、#2336 B=4の28.64 msにも今回の28.76 msが近い。B=1のPT16は #2303 の27.197 msに対して20.49 msだった。先行runは236 cropを使い、比較には入力集合や実行時設定の違いがある。 [ViT単体](000077-run-i2303-vit-tensorrt.md) · [batch掃引](000080-run-i2336-vision-batch-sweep.md)

HMR2の内部16 cropのforward中央値はPT32/PT16/TRT16が390.10/125.09/55.13 msで、#2336 の410.37/145.81/55.07 msと同じ順位だった。身体合計は毎回HMR2を実行し、GVHMRの16フレームを復号する。#2355の145/177フレームのdenoiser試行は事前に固定した画像特徴を使い、カメラ座標の関節P95は8.78/8.99 mmだった。今回の21.79 mmとの比較には系列長と入力特徴の作り方の違いがある。 [HMR2 batch掃引](000080-run-i2336-vision-batch-sweep.md) · [GVHMR単体](000082-run-i2355-gvhmr-pred-cam-clamp.md)

実験は2026-11-24から25日まで、driver 621.24、PyTorch 2.13.0+cu130、CUDA 13.0、TensorRT 10.14.1.48.post1で行った。電力上限は180 W。load直前のclock・電力、torchとNVMLのサンプル、loadと初回推論、重みとplanの全SHA-256を条件別に保存した。ViTPoseのflipなしB=1向けのbuildは74.35秒で、保存済みONNXを入力とした。 [生の計時値と再現手順](../../runs/run-person-model-speed-baseline/README.md) · [queueとplanの記録](../../runs/run-person-model-speed-baseline/queue-receipts.json)
