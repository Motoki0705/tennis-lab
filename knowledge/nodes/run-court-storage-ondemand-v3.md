---
id: run-court-storage-ondemand-v3
type: run
title: コート学習と常駐描画の基本速度比較
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
  cached_gpu_images_per_second: 19.216784610990725
  npy_loader4_images_per_second: 18.47536470137141
  npy_pipeline_images_per_second: 12.425122103138998
  compressed_pipeline_images_per_second: 8.653266556636492
  ondemand_serial_images_per_second: 8.91333674660916
  ondemand_prefetch_images_per_second: 8.990177460510763
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/benchmark_court_ondemand.py
    --repo /home/kamimura/projects/tennis-lab --nht-worker /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand/scripts/resident_render_worker.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --output outputs/storage-ondemand/benchmark-v3
artifacts:
  run_dir: knowledge/runs/run-court-storage-ondemand-v3
  metrics: knowledge/runs/run-court-storage-ondemand-v3/metrics.json
  nht_patch: knowledge/runs/run-court-storage-views-v1/nht-resident.patch
parents:
- run-court-storage-ondemand-v2
relations: []
tags:
- court
- storage
- ondemand
- throughput
---

## 考察 / Findings

### 要約
保存RGBと常駐NHT RGBは完全一致したが、単純なオンデマンド経路は4-worker DataLoaderより遅い。

### アーキテクチャ詳細
RTX 5060 Ti、B00、64カメラ、batch 8、各16 step。DINOv3 ViT-B/16＋8層transformer＋DPT＋KP/SEG/LINE、LoRA、bf16、AdamW。入力はpose_safe評価変換（short-side 256（align後256×456））。compileは無効。固定教師、ランダム画像augmentationとLightningログ処理は測定外。

### メトリクスの解釈
images_per_secondは各mode全stepの画像数/壁時計時間。cached_gpuはデータ処理を除いた上限。loss値は性能実験中の値で、収束や精度改善の根拠ではない。

### アーキテクチャ⇄メトリクスの因果考察
render-onlyでは約0.18秒/8枚だが、実経路にはCPU処理とプロセス境界も必要。GPUだけが原因とは分離できない。

### 既存実験との比較
v2のcollate不具合を修正。npy_pipelineとcompressed_pipelineはworker=0であり、通常のDataLoaderとの区別が必要。

### 次に有効な実験
圧縮データの4-worker読み込み、CPU処理までの先読み、レンダリング画像の再利用を評価する。
