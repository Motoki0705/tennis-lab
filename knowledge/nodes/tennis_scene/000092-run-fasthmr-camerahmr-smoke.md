---
id: run-fasthmr-camerahmr-smoke
type: run
task: tennis_scene
sequence: 92
recorded_at: '2026-11-25'
title: 固定テニス映像でのFastHMR CameraHMR実重みPT32・PT16の動作確認
issue: 2491
provider: codex
session: 01a20d45-575e-7a53-90c5-48b09053fa2b
date: '2026-11-25'
status: done
config:
  model: fasthmr-camerahmr
  data: meiji_4cam/multiview-player-evaluation-v1
  feature_batch_size: 4
  diffusion_seed: 0
metrics:
  camera_coco17_delta_p95_mm: 45.1199356110281
  camera_vertices_delta_p95_mm: 45.18953297772461
  camera_translation_delta_p95_mm: 42.98894642696802
summary_impact: none
repro:
  commit: ede444fd7a1b0b33d4624bc2be86109a1bcba4e2
  branch: codex/parent-2492-issue-2491-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash outputs/fasthmr-setup/run_gpu_smoke.sh
artifacts:
  run_dir: knowledge/runs/run-fasthmr-camerahmr-smoke
  log: .training_queue/repro/0-8204473201027913525_2026-11-24T13-26-38-972086474Z_2037423_i2491-fasthmr-pt32-pt16-smoke/queue.log
  log_checksum: knowledge/runs/run-fasthmr-camerahmr-smoke/queue-log.sha256.txt
parents: [run-meiji-multiview-player-evaluation]
relations: []
papers: []
tags: [fasthmr, body_recovery]
---

## 考察 / Findings

#2495のbody入力をSHA-256で照合し、`video_000/clip_003` の選手0に対して、選択カメラcam0のsource frame 0・2・4・6・8・10・12・14を推論した。コホートmanifestは `3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580`。実行HEADは `ede444fd7a1b0b33d4624bc2be86109a1bcba4e2`。RTX 5060 Ti、PyTorch 2.13.0+cu130、CUDA 13.0で、TF32をオフ、特徴抽出batch 4、diffusion seed 0とした。CameraHMRとFastHMRの重み・SMPL資産は実装で照合した。 [固定入力](000084-run-meiji-multiview-player-evaluation.md) [実行記録](../../runs/run-fasthmr-camerahmr-smoke/run.json)

PT32とPT16はどちらも8フレーム分の有限な69次元SMPL姿勢、6890頂点とCOCO17関節を生成した。2つの生予測のSHA-256とタイムラインを読み戻し、カメラ座標のユークリッド距離を集計した。

| 比較対象 | 件数 | 中央値 mm | P95 mm |
| --- | ---: | ---: | ---: |
| COCO17関節 | 136 | 15.76 | 45.12 |
| SMPL頂点 | 55,120 | 15.78 | 45.19 |
| カメラ並進 | 8 | 15.39 | 42.99 |

このサンプルの関節差は、親 #2492 の基準値10 mmを上回った。この試行は1選手の部分区間に限る。後続の代表6 clipでは同期比較用のPT16予測を保存し、PT32とのparityは未判定とした。置き換えの見送りは後続の速度計測に基づく。 [照合したSHAと診断値](../../runs/run-fasthmr-camerahmr-smoke/observations.json) [代表6 clipの記録](000094-run-fasthmr-camerahmr-pt16-review.md) [採否と速度](000095-run-fasthmr-camerahmr-pt16-speed.md)

## 再現

元コマンドと固定HEADは [run.json](../../runs/run-fasthmr-camerahmr-smoke/run.json) と [キューが生成した再現スクリプト](../../runs/run-fasthmr-camerahmr-smoke/repro.queue.sh) に保存した。再実行時は `TENNIS_REPO` に外部checkout・利用者取得資産・依存を用意した固定HEADのworktreeを指定し、同梱の [repro.sh](../../runs/run-fasthmr-camerahmr-smoke/repro.sh) を共有GPUキュー経由で呼ぶ。生予測は元ジョブの `.training_queue/repro/<job>/predictions/` にあり、[verify_predictions.py](../../runs/run-fasthmr-camerahmr-smoke/verify_predictions.py) で照合できる。
