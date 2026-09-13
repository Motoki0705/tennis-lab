---
id: run-court-storage-ondemand-v5
type: run
title: 通常の256角学習augmentationを含むオンデマンド比較
provider: codex
session: 01a0985c-eb87-7310-9657-5411e2818d4e
date: '2026-09-13'
status: done
config:
  model: DINOv3 ViT-B/16 + 8-layer transformer + DPT + LoRA
  loss: KP/SEG/LINE default
  data: B00 batch8, 256x256 training augmentation, bf16, compile=false
metrics:
  cached_gpu_images_per_second: 31.5856181573624
  npy_loader4_images_per_second: 30.833403422596813
  compressed_loader4_images_per_second: 30.21009207745778
  npy_pipeline_images_per_second: 18.347441174429783
  compressed_pipeline_images_per_second: 11.282014857752957
  ondemand_serial_images_per_second: 10.796824002971782
  ondemand_prefetch_images_per_second: 12.462426726832401
  ondemand_cpu_prefetch_images_per_second: 11.260644667515352
  ondemand_reuse4_images_per_second: 22.683924099600524
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/benchmark_court_ondemand.py
    --repo /home/kamimura/projects/tennis-lab --nht-worker /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand/scripts/resident_render_worker.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --output outputs/storage-ondemand/benchmark-v5
    --steps 24 --geometry training
artifacts:
  run_dir: knowledge/runs/run-court-storage-ondemand-v5
  metrics: knowledge/runs/run-court-storage-ondemand-v5/metrics.json
  nht_patch: knowledge/runs/run-court-storage-views-v1/nht-resident.patch
parents:
- run-court-storage-ondemand-v4
relations: []
tags:
- court
- storage
- ondemand
- throughput
---

## 考察 / Findings

### 要約
通常の256×256学習batchと画像augmentationを使っても結論は同じ。4-worker可逆圧縮は未圧縮とほぼ同等、毎batch新規レンダリングは大きく遅く、4回再利用で一部回復する。

### アーキテクチャ詳細
v4の評価用変換（256×456）から、実際のtrain pipelineへ変更した。pose_safe preset、pose loss無効のため256×256のcrop/resizeと学習用color jitter/blurを適用する。LoRA・KP/SEG/LINE、batch8、bf16、AdamW、compile無効、24 step/mode。Python/NumPy/Torchのseedは42。各modeは順に実行して重みを引き継ぐ。rendererは常駐し、CPU受渡しは/dev/shmのNPY。

### メトリクスの解釈
基準はnpy_loader4。短時間・warm page cacheの計測なので圧縮の約2%差を恒常的な劣化と断定しない。cached_gpuとreuse4は準備済みGPU batchを繰り返すため、毎stepのaugmentationを再生成しない。reuse4の描画は6 batch、循環index上の異なる画像は16種類であり、学習画像数/秒と新規画像数/秒は別。損失は速度測定中の値であり、汎化精度を示さない。

### アーキテクチャ⇄メトリクスの因果考察
入力を小さくすると学習自体が速くなり、959×539のレンダリングやCPU前処理の比率が高くなる。非同期化だけでは同一GPUの計算量・context共有・CPU処理を隠し切れなかった。各原因の寄与は分離していない。

### 既存実験との比較
v4と入力サイズとaugmentationが異なるため、絶対速度の直接比較にはこの差を含む。両方で圧縮4-workerが未圧縮に近く、毎batch生成が遅くなる傾向は共通した。最終提案の速度表はこのv5を主とする。

### 次に有効な実験
容量上限のraw uint8 cacheを用いて、多数cameraをシャッフルしながらaugmentationを毎回適用する方式。compile有効、mixed-source、長期精度・time-to-qualityの評価を行う。
