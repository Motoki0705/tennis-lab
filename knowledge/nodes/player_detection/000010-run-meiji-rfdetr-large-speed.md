---
id: run-meiji-rfdetr-large-speed
type: run
task: player_detection
sequence: 10
recorded_at: '2026-11-25'
title: Meiji 4cam 固定画像での RF-DETR-L と DINO の速度・VRAM比較
issue: 2487
provider: codex
date: '2026-11-25'
status: done
summary_impact: none
config:
  model: rfdetr-l
  checkpoint: rf-detr-large-2026.pth
  baseline: dino-swin-l
  input_id: meiji_4cam/person-model-speed-v1
  input_manifest_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
  timing_protocol_sha256: 39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2
  backends: [pt16, trt16]
  batch_sizes: [1, 2]
  warmup: 30
  iterations_per_block: 100
  blocks: 3
  torch_cpu_threads: 4
  opencv_cpu_threads: 1
  gpu: NVIDIA GeForce RTX 5060 Ti
  tf32: false
  parity_status: PT32未計測
metrics:
  fixed_images: 96
  timed_batches: 1200
  stage_wall_p50_ms_per_image_pt16_b1: 39.7737985
  stage_wall_p50_ms_per_image_pt16_b2: 27.40488525
  stage_wall_p50_ms_per_image_trt16_b1: 26.4302925
  stage_wall_p50_ms_per_image_trt16_b2: 24.1442045
  nvml_peak_delta_mib_pt16_b1: 368
  nvml_peak_delta_mib_pt16_b2: 442
  nvml_peak_delta_mib_trt16_b1: 142
  nvml_peak_delta_mib_trt16_b2: 158
repro:
  commit: 8875cd0c17459660ff4f9f491c3a5fb1b965bfb5
  branch: codex/parent-2492-issue-2487-rfdetr
  harness_merge_commit: 59faa32a28d2f54e6492abdba2429846fa0397e3
artifacts:
  run_dir: knowledge/runs/run-meiji-rfdetr-large-speed
  results: knowledge/runs/run-meiji-rfdetr-large-speed/results.json
  comparison: knowledge/runs/run-meiji-rfdetr-large-speed/comparison.csv
  output_dir: .worktree-outputs/parent-2492-issue-2487-rfdetr/player-model-comparison/speed-synchronized-v2/rfdetr-l
parents:
- run-meiji-rfdetr-large-accuracy
- run-person-model-speed-low-precision
relations: []
papers: []
tags:
- meiji-4cam
- rf-detr
- fp16
- tensorrt
---

## 速度とVRAM

RTX 5060 Ti上で、RF-DETR-LとDINO 4-scale Swin-Lに同じ6 clipの固定96画像を同じ順で入力した。PyTorchのCPUスレッドを4、OpenCVを1に揃え、専用CUDA streamで各条件を30回warm-upし、100反復を3ブロック計測した。RF-DETR-Lの4条件すべてが完了した。parityは2026-11-25のユーザー判断に従い「PT32未計測」とした。 [固定入力と計時規約](../../runs/run-person-model-speed-baseline/README.md) · [DINOの基準値](../tennis_scene/000086-run-person-model-speed-low-precision.md)

wallは前後処理・転送・forward・バッチ終端の同期を含む。VRAM増分は、モデルload直前から計測終了までにNVMLで採ったデバイス使用量の最大値とload直前の差。環境とモデルGPU時間、スループット、PyTorch allocatorのピークは条件別記録にある。 [速度と精度の全指標](../../runs/run-meiji-rfdetr-large-speed/comparison.csv) · [環境と測定ファイル](../../runs/run-meiji-rfdetr-large-speed/results.json)

| Backend | Batch | RF-DETR-L wall P50 / P95 (ms/画像) | DINO wall P50 / P95 (ms/画像) | RF-DETR-L VRAM増分 (MiB) | DINO VRAM増分 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PT16 | 1 | 39.77 / 48.48 | 165.69 / 170.71 | 368 | 2,406 |
| PT16 | 2 | 27.40 / 41.15 | 162.36 / 167.80 | 442 | 3,290 |
| TRT16 | 1 | 26.43 / 33.46 | 94.79 / 101.53 | 142 | 984 |
| TRT16 | 2 | 24.14 / 52.00 | 85.91 / 94.22 | 158 | 996 |

batch 1のwall P50はDINOよりPT16で76.0%、TRT16で72.1%短かった。RF-DETR-LのTRT16はbatch 2でP50が下がり、P95は33.46から52.00 ms/画像へ上がった。実測時間全体でのスループットはbatch 1が36.80画像/s、batch 2が35.64画像/sだった。 [反復単位のPT16データ](../../runs/run-meiji-rfdetr-large-speed/measurements/pt16.csv) · [反復単位のTRT16データ](../../runs/run-meiji-rfdetr-large-speed/measurements/trt16.csv)

## 疑似GTとの一致度

共通の6,844画像・26,276選手BBoxを、両モデルとも同じ採点器で評価した保存済みbatch 1の結果を用いた。検出のscore下限はRF-DETR-Lが0.25、DINOが0.30。主指標のRecall@0.50はPT16で96.82%と96.69%、TRT16で96.81%と96.67%だった。Recall@0.75はPT16で42.03%と46.99%、TRT16で42.09%と47.07%だった。IoU 0.75での結果は疑似BBoxの余白に敏感で、今回の比較では各辺10%に固定した。batch 2の疑似GT採点と、ViTPose・追跡・3D再構成まで通した品質は未検証。 [精度と余白感度](000009-run-meiji-rfdetr-large-accuracy.md) · [精度と速度の条件別一覧](../../runs/run-meiji-rfdetr-large-speed/comparison.csv)

## 検証と再現

4件のqueue完了記録と実ファイルを照合した。RF-DETR-Lと対応するDINOの各300反復について、画像ID順、ブロック番号とP50/P95・スループットを生サンプルから検証した。RF-DETR-Lの実行HEADは `8875cd0c17459660ff4f9f491c3a5fb1b965bfb5`。TRT16 batch 1のplanは保存済み精度計測と同じSHA-256だった。batch 2のplanは初回速度計測で作成した。batch 1の2レポートは #2487 の統合集計で両方受理されることを確認した。 [実行コマンド・成果物・SHA-256](../../runs/run-meiji-rfdetr-large-speed/results.json) · [再現手順](../../runs/run-meiji-player-model-comparison/rfdetr.md#fixed-speed-comparison) · [CPUスレッド数を揃える前の記録](../../runs/run-meiji-player-model-comparison/rfdetr-speed-initial.json)

人物検出単体の比較を追加した。運用判断に必要な下流品質は未検証のため、[人物の現状判断](../../summary.md#人物player_detectionperson_trackingplayer_association)への影響は `none` とした。
