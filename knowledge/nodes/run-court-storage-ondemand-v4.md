---
id: run-court-storage-ondemand-v4
type: run
title: 圧縮4-worker・先読み・描画再利用の比較
provider: codex
session: 01a0985c-eb87-7310-9657-5411e2818d4e
date: '2026-09-13'
status: done
config:
  model: DINOv3 ViT-B/16 + transformer + DPT + LoRA
  loss: kp_seg_line default
  data: B00, batch8, pose_safe, bf16, compile=false
metrics:
  rerender_float_mae: 0.0
  cached_gpu_images_per_second: 18.972394246177238
  npy_loader4_images_per_second: 18.64369885648737
  compressed_loader4_images_per_second: 18.95687101285919
  npy_pipeline_images_per_second: 12.396603640096258
  compressed_pipeline_images_per_second: 8.83392978974099
  ondemand_serial_images_per_second: 7.066292200672074
  ondemand_prefetch_images_per_second: 8.916670424697589
  ondemand_cpu_prefetch_images_per_second: 8.38211949016401
  ondemand_reuse4_images_per_second: 13.769175382981723
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/benchmark_court_ondemand.py
    --repo /home/kamimura/projects/tennis-lab --nht-worker /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand/scripts/resident_render_worker.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --output outputs/storage-ondemand/benchmark-v4
    --steps 24
artifacts:
  run_dir: knowledge/runs/run-court-storage-ondemand-v4
  metrics: knowledge/runs/run-court-storage-ondemand-v4/metrics.json
  nht_patch: knowledge/runs/run-court-storage-views-v1/nht-resident.patch
parents:
- run-court-storage-ondemand-v3
relations: []
tags:
- court
- storage
- ondemand
- throughput
---

## 考察 / Findings

### 要約
可逆圧縮は4-workerで読み込めば未圧縮と同等の速度。毎batch新規レンダリングは大きく減速し、同一batchの4回再利用で一部回復した。

### アーキテクチャ詳細
v3と同じモデル・データ・最適化条件で24 step/mode。compressed_loader4は4 worker。ondemand_cpu_prefetchはworkerへの要求とCPU前処理を別threadへ移す。ondemand_reuse4は1回の描画batchを4回学習し、新規描画要求を1/4にする（24 step中6回、実装の循環indexにより新規画像は16種類）。全modeは順に実行し、モデル重みは引き継ぐ。

### メトリクスの解釈
画像/秒は学習処理数。再利用方式の「新規画像/秒」はこの値とは異なる。全方式でrendererは同時に常駐しているがbaselineでは描画しない。各modeのGPU peakは学習processのallocated値で、rendererやCUDA contextを含むGPU総使用量ではない。

### アーキテクチャ⇄メトリクスの因果考察
render-only時間と学習時間を単純にmaxで重ねられない。GPU context競合、CPU前処理、NPYの共有メモリ受渡しを含む測定であり、GPU競合だけの増分は未分離。圧縮のCPUコストはDataLoaderの並列先読みにより隠れたと解釈できる。

### 既存実験との比較
v3と同じ傾向。4-worker圧縮は速度低下を示さない。同期・非同期の差は小規模な単一runなので大きな優劣を断定しない。

### 次に有効な実験
固定容量のuint8画像cacheを用いたmicro-epoch生成・学習の交互実行。compile有効、本来のaugmentation、複数sceneと独立testを使う収束評価は未実施。
