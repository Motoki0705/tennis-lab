---
id: run-person-stage-precision-smoke-20261118
type: run
task: tennis_scene
sequence: 48
recorded_at: '2026-11-18'
title: Meiji 4camの固定8画像で人物推論のFP32一致と混合精度を確認
issue: 2238
provider: codex
session: 01a1ec52-c2ef-70e3-bd48-588f68ed717e
date: '2026-11-18'
status: done
config:
  clips:
  - clip_037
  - clip_000
  cameras_per_clip: 4
  dino_batch_size: 1
  vitpose_flip_default: true
  vitpose_batch_size: 16
  torch: 2.13.0+cu130
  gpu: NVIDIA GeForce RTX 5060 Ti
metrics:
  dino_fp32_exact: true
  vitpose_fp32_exact: true
  images: 8
  detection_rows: 63
  dino_bf16_seconds: 1.6648
  dino_fp16_seconds: 1.5193
  dino_fp32_seconds: 2.3966
  vitpose_bf16_seconds: 1.2129
  vitpose_fp16_seconds: 1.14
  vitpose_fp32_seconds: 3.1203
  vitpose_no_flip_seconds: 1.6071
repro:
  commit: 5854fa629ac8891ae9210d553c72e3c9ca5a9213
  branch: codex/person-stage-speed
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so
    .venv/bin/python -u knowledge/runs/run-person-stage-precision-meiji-20261118/gpu_guard.py
    --health outputs/precision/smoke/health knowledge/runs/run-person-stage-precision-meiji-20261118/smoke.py
    --dataset data/tennis_multiview/processed/meiji_4cam/dataset --pin knowledge/runs/run-person-stage-precision-meiji-20261118/smoke-inputs.json
    --output outputs/precision/smoke/metrics.json
artifacts:
  run_dir: knowledge/runs/run-person-stage-precision-smoke-20261118
  log: .training_queue/logs/0-8205032956589656196_2026-11-18T01-57-23-410343803Z_2135948_i2238-precision-smoke.log
  output_dir: .worktree-outputs/person-stage-speed/precision/smoke
parents: []
relations: []
papers: []
tags:
- inference-precision
- person-detection
- pose-estimation
summary_impact: none
---

# 固定8画像での確認

Meiji 4camの clip_037（frame 146）と clip_000（frame 178）を4カメラずつ比較した。
DINOのbox・scoreとViTPoseのCOCO17座標・heatmap peakは、変更前のFP32と変更後の既定FP32で全要素が一致した。
ViTPoseには同じ63件の検出boxを渡し、cropとUDP復号を揃えた。
[入力と重みの固定情報](../../runs/run-person-stage-precision-meiji-20261118/smoke-inputs.json)

| モデル | 条件 | 8画像の推論時間（秒） | FP32比 | PyTorch割当ピーク（MiB） |
|---|---|---:|---:|---:|
| ViTPose | FP32 | 3.120 | 1.00× | 2772 |
| ViTPose | BF16 | 1.213 | 2.57× | 3932 |
| ViTPose | FP16 | 1.140 | 2.74× | 3903 |
| ViTPose | FP32・flipなし | 1.607 | 1.94× | 2654 |
| DINO | FP32 | 2.397 | 1.00× | 1780 |
| DINO | BF16 | 1.665 | 1.44× | 1685 |
| DINO | FP16 | 1.519 | 1.58× | 1684 |

各条件でウォームアップを1回行い、モデルの読込・動画のdecodeを除いた。DINOはtensor転送とbox・scoreの復号を、
ViTPoseはcrop・GPU推論・UDP復号を含む同期wall timeを計測した。
DINOのBF16・FP16で8画像それぞれの検出数は一致した。box・scoreの差は全クリップの計測で求める。
ViTPoseのxy成分最大差はBF16で19.46 px、FP16で11.78 px、flipなしで64.87 px。
BF16・FP16は同じcrop数でもFP32より割当ピークが約1.1 GiB高かった。
[計測結果](../../runs/run-person-stage-precision-smoke-20261118/metrics.json)

FP32との比較に使ったフレームは各カメラで1枚。標準の採用精度に関する判断は維持する（summary_impact: none）。
全フレームでの検出、追跡、人物対応、三角測量の差は別runで計測する。
[計測コード](../../runs/run-person-stage-precision-meiji-20261118/smoke.py)
