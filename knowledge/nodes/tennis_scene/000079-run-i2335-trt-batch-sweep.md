---
id: run-i2335-trt-batch-sweep
type: run
task: tennis_scene
sequence: 79
recorded_at: '2026-11-21'
title: CLIP-ReIDとGVHMRのバッチサイズ別推論比較
issue: 2335
provider: codex
date: '2026-11-21'
status: running
summary_impact: none
config:
  dataset: meiji_4cam
  evaluation_clips: [clip_000, clip_037]
  accuracy_reference: PyTorch FP32 batch 1
  clipreid_batches: [1, 4, 8, 16, 32]
  clipreid_eval_crops: 236
  clipreid_ptq_judgement_crops: 224
  clipreid_trt_min_opt_max: [1, 16, 32]
  gvhmr_batches: [1, 2, 4]
  gvhmr_eval_lengths: [145, 177]
  gvhmr_trt_min_opt_max_per_length: [1, 4, 4]
metrics:
  timing_accuracy_nsys_cases: 32
  planned_cases: 43
  ncu_with_matching_nsys: 6
  clipreid_b16:
    fp32_images_per_second: 356.37
    pytorch_fp16_images_per_second: 1191.40
    trt_fp16_images_per_second: 2358.36
    trt_int8_images_per_second: 2372.23
  gvhmr_b4:
    l145_fp32_sequences_per_second: 242.92
    l177_fp32_sequences_per_second: 232.84
    l145_fp16_incam_joint_p95_mm: 8.52
    l177_fp16_incam_joint_p95_mm: 8.64
artifacts:
  run_dir: knowledge/runs/run-i2335-trt-batch-sweep
  metrics: knowledge/runs/run-i2335-trt-batch-sweep/metrics.json
  validation: knowledge/runs/run-i2335-trt-batch-sweep/validation.json
parents:
- run-i2297-clipreid-tensorrt
- run-i2303-vit-tensorrt
- run-i2305-gvhmr-tensorrt
relations: []
papers: []
tags: []
---

## 結果

CLIP-ReIDはFP32、PyTorch FP16、TensorRT FP16、INT8の全5バッチで速度・VRAM・nsys・単体精度を得た。FP16とINT8のTensorRTはB=16でそれぞれ2,358、2,372 crop/sで、B=32では2,341、2,283 crop/sだった。INT8はPTQ設定選定用12件を除いた224件で全バッチが暫定許容差内だった。FP16・INT8のplanは両端とoptで主要演算の精度を照合し、B=0と33を終了値1で拒否した。 [各ケースの集計](../../runs/run-i2335-trt-batch-sweep/metrics.json)・[実行層と境界の検証](../../runs/run-i2335-trt-batch-sweep/validation.json)

GVHMRはPyTorchの2精度で、評価系列長ごとに全3バッチを測定した。FP32 B=4は145フレームで242.92系列/s、177フレームで232.84系列/sだった。FP32のバッチ間のカメラ座標COCO-17関節差はP95最大0.017 mmだった。FP16のカメラ座標関節差はP95 8.45–8.65 mm、重力整列座標では16.47–24.05 mmだった。FP16の単体結果は座標系によって暫定の10 mm基準をまたぐ。全ケースで長さとmaskが一致した。 [精度と速度の集計](../../runs/run-i2335-trt-batch-sweep/metrics.json)

代表バッチのncuはCLIP-ReIDのFP32・INT8と、GVHMRのPyTorch 2精度・2系列長でnsysとkernel件数が一致した。CLIP-ReIDのFP32のDRAM readはB=1の0.484 GBからB=16で3.197 GBへ増え、1 crop当たりは0.484 GBから0.200 GBへ減った。INT8は0.155 GBから0.974 GBで、1 crop当たりは0.155 GBから0.061 GBだった。この比較には、TensorRTのB=1専用planと可変batch planのタクティクスの差が含まれる。 [B=1の記録と代表バッチの指標](../../runs/run-i2335-trt-batch-sweep/metrics.json)

## 残る計測

CLIP-ReIDのFP8はQ/DQが形状参照にも接続されたためビルド前に停止した。FP16のPyTorchとTensorRTのncuはそれぞれ119/189、66/114 kernelで時間切れになった。GVHMRではBと系列長を同時に可変にしたTensorRTビルドがSIGSEGVで停止した。FP8の形状参照を修正したグラフ、ncuの残りのkernel、GVHMRの系列長別の可変batch planで確認を続ける。 [失敗と再試行条件](../../runs/run-i2335-trt-batch-sweep/validation.json)・[再現手順](../../runs/run-i2335-trt-batch-sweep/README.md)

現時点の結果を追加の単体実測として記録する。標準パイプライン構成の判断は、残るTensorRT計測と下流E2Eを含めて行う。
