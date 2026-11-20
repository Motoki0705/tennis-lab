---
id: run-i2305-gvhmr-tensorrt
type: run
task: tennis_scene
sequence: 76
recorded_at: '2026-11-20'
title: GVHMRの可変長PyTorch・TensorRT推論を5条件で比較
issue: 2305
provider: codex
date: '2026-11-20'
status: done
config:
  checkpoint: gvhmr/gvhmr_siga24_release.ckpt
  dataset: meiji_4cam
  eval_clips:
  - clip_000
  - clip_037
  calibration_clip: clip_027
  batch: 1
  dynamic_profile_lengths:
    min: 117
    opt: 145
    max: 177
  calibration_method: max
  calibration_eps:
  - cpu
  precision_protocol_issue: 2297
metrics:
  l145:
    pytorch_fp32:
      median_ms: 18.942
      p95_ms: 25.6031
      sequences_per_second: 50.913
      frames_per_second: 7382.411
      nvml_peak_sampled_bytes: 683773952
      torch_peak_allocated_bytes: 215580160
      nsys_gpu_kernels: 730
      dram_read_bytes: 275522304
      dram_write_bytes: 0
      top_stall_reason: mio_throttle
      top_stall_pct_time_weighted: 21.4807
      tensor_active_pct_time_weighted: 0.0
      body_pose_deg_p95: 0.0
      betas_abs_p95: 0.0
      incam_global_orient_deg_p95: 0.0
      incam_transl_l2_mm_p95: 0.0
      incam_coco17_l2_mm_p95: 0.0
      global_global_orient_deg_p95: 0.0
      global_transl_l2_mm_p95: 0.0
      global_coco17_l2_mm_p95: 0.0
      matching_length_mask_frames: 580
    pytorch_fp16:
      median_ms: 14.6502
      p95_ms: 16.8719
      sequences_per_second: 67.45
      frames_per_second: 9780.32
      nvml_peak_sampled_bytes: 585207808
      torch_peak_allocated_bytes: 125176832
      nsys_gpu_kernels: 701
      dram_read_bytes: 106659072
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 18.5951
      tensor_active_pct_time_weighted: 7.298
      body_pose_deg_p95: 0.06154
      betas_abs_p95: 0.001076
      incam_global_orient_deg_p95: 0.04072
      incam_transl_l2_mm_p95: 8.50933
      incam_coco17_l2_mm_p95: 8.65433
      global_global_orient_deg_p95: 0.39412
      global_transl_l2_mm_p95: 23.85448
      global_coco17_l2_mm_p95: 24.04985
      matching_length_mask_frames: 580
    tensorrt_fp16:
      median_ms: 2.5085
      p95_ms: 3.1701
      sequences_per_second: 393.57
      frames_per_second: 57067.682
      nvml_peak_sampled_bytes: 549556224
      torch_peak_allocated_bytes: 363520
      nsys_gpu_kernels: 135
      dram_read_bytes: 71316992
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 14.2485
      tensor_active_pct_time_weighted: 20.1502
      body_pose_deg_p95: 0.08652
      betas_abs_p95: 0.001615
      incam_global_orient_deg_p95: 0.0651
      incam_transl_l2_mm_p95: 8.4903
      incam_coco17_l2_mm_p95: 8.61373
      global_global_orient_deg_p95: 0.42322
      global_transl_l2_mm_p95: 23.2429
      global_coco17_l2_mm_p95: 23.49857
      matching_length_mask_frames: 580
    tensorrt_int8:
      median_ms: 2.3321
      p95_ms: 3.1707
      sequences_per_second: 411.619
      frames_per_second: 59684.726
      nvml_peak_sampled_bytes: 518098944
      torch_peak_allocated_bytes: 724480
      nsys_gpu_kernels: 150
      dram_read_bytes: 39428352
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 13.7943
      tensor_active_pct_time_weighted: 11.2592
      body_pose_deg_p95: 1.84789
      betas_abs_p95: 0.058501
      incam_global_orient_deg_p95: 0.8485
      incam_transl_l2_mm_p95: 92.08809
      incam_coco17_l2_mm_p95: 94.54001
      global_global_orient_deg_p95: 1.11018
      global_transl_l2_mm_p95: 84.0148
      global_coco17_l2_mm_p95: 84.1068
      matching_length_mask_frames: 580
    tensorrt_fp8:
      median_ms: 2.3282
      p95_ms: 3.0988
      sequences_per_second: 411.768
      frames_per_second: 59706.362
      nvml_peak_sampled_bytes: 516001792
      torch_peak_allocated_bytes: 724480
      nsys_gpu_kernels: 151
      dram_read_bytes: 40694016
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 11.9792
      tensor_active_pct_time_weighted: 10.1338
      body_pose_deg_p95: 1.60641
      betas_abs_p95: 0.034716
      incam_global_orient_deg_p95: 0.61204
      incam_transl_l2_mm_p95: 35.55534
      incam_coco17_l2_mm_p95: 36.9482
      global_global_orient_deg_p95: 0.65056
      global_transl_l2_mm_p95: 47.21098
      global_coco17_l2_mm_p95: 47.60508
      matching_length_mask_frames: 580
  l177:
    pytorch_fp32:
      median_ms: 16.5408
      p95_ms: 18.8704
      sequences_per_second: 59.482
      frames_per_second: 10528.333
      nvml_peak_sampled_bytes: 685871104
      torch_peak_allocated_bytes: 216482304
      nsys_gpu_kernels: 867
      dram_read_bytes: 345229056
      dram_write_bytes: 0
      top_stall_reason: mio_throttle
      top_stall_pct_time_weighted: 22.5104
      tensor_active_pct_time_weighted: 0.0
      body_pose_deg_p95: 0.0
      betas_abs_p95: 0.0
      incam_global_orient_deg_p95: 0.0
      incam_transl_l2_mm_p95: 0.0
      incam_coco17_l2_mm_p95: 0.0
      global_global_orient_deg_p95: 0.0
      global_transl_l2_mm_p95: 0.0
      global_coco17_l2_mm_p95: 0.0
      matching_length_mask_frames: 708
    pytorch_fp16:
      median_ms: 14.1154
      p95_ms: 16.4581
      sequences_per_second: 69.196
      frames_per_second: 12247.648
      nvml_peak_sampled_bytes: 587304960
      torch_peak_allocated_bytes: 125765632
      nsys_gpu_kernels: 732
      dram_read_bytes: 150749696
      dram_write_bytes: 512
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 20.2317
      tensor_active_pct_time_weighted: 6.4098
      body_pose_deg_p95: 0.06287
      betas_abs_p95: 0.001378
      incam_global_orient_deg_p95: 0.04653
      incam_transl_l2_mm_p95: 8.56457
      incam_coco17_l2_mm_p95: 8.55025
      global_global_orient_deg_p95: 0.19582
      global_transl_l2_mm_p95: 20.3489
      global_coco17_l2_mm_p95: 20.12669
      matching_length_mask_frames: 708
    tensorrt_fp16:
      median_ms: 2.498
      p95_ms: 2.9789
      sequences_per_second: 394.235
      frames_per_second: 69779.593
      nvml_peak_sampled_bytes: 549556224
      torch_peak_allocated_bytes: 444416
      nsys_gpu_kernels: 135
      dram_read_bytes: 71583232
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 14.242
      tensor_active_pct_time_weighted: 19.3265
      body_pose_deg_p95: 0.08495
      betas_abs_p95: 0.001844
      incam_global_orient_deg_p95: 0.06929
      incam_transl_l2_mm_p95: 8.68232
      incam_coco17_l2_mm_p95: 8.7207
      global_global_orient_deg_p95: 0.18055
      global_transl_l2_mm_p95: 18.55338
      global_coco17_l2_mm_p95: 18.32453
      matching_length_mask_frames: 708
    tensorrt_int8:
      median_ms: 2.3785
      p95_ms: 3.0136
      sequences_per_second: 409.651
      frames_per_second: 72508.199
      nvml_peak_sampled_bytes: 518098944
      torch_peak_allocated_bytes: 884224
      nsys_gpu_kernels: 150
      dram_read_bytes: 40538624
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 12.636
      tensor_active_pct_time_weighted: 10.9954
      body_pose_deg_p95: 2.11569
      betas_abs_p95: 0.040464
      incam_global_orient_deg_p95: 1.09808
      incam_transl_l2_mm_p95: 80.19242
      incam_coco17_l2_mm_p95: 81.96247
      global_global_orient_deg_p95: 1.22494
      global_transl_l2_mm_p95: 49.98961
      global_coco17_l2_mm_p95: 51.85314
      matching_length_mask_frames: 708
    tensorrt_fp8:
      median_ms: 2.5122
      p95_ms: 3.0634
      sequences_per_second: 391.694
      frames_per_second: 69329.829
      nvml_peak_sampled_bytes: 516001792
      torch_peak_allocated_bytes: 884224
      nsys_gpu_kernels: 151
      dram_read_bytes: 42071040
      dram_write_bytes: 0
      top_stall_reason: long_scoreboard
      top_stall_pct_time_weighted: 11.6236
      tensor_active_pct_time_weighted: 9.9489
      body_pose_deg_p95: 1.81467
      betas_abs_p95: 0.050989
      incam_global_orient_deg_p95: 0.71728
      incam_transl_l2_mm_p95: 67.2482
      incam_coco17_l2_mm_p95: 68.93021
      global_global_orient_deg_p95: 0.80757
      global_transl_l2_mm_p95: 71.86839
      global_coco17_l2_mm_p95: 71.959
      matching_length_mask_frames: 708
artifacts:
  run_dir: knowledge/runs/run-i2305-gvhmr-tensorrt
  output_dir: .worktree-outputs/trt-gvhmr/trt-gvhmr
  metrics: knowledge/runs/run-i2305-gvhmr-tensorrt/metrics.json
  validation: knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json
parents:
- run-i2297-clipreid-tensorrt
relations: []
papers: []
tags:
- gvhmr
- tensorrt
- ptq
summary_impact: none
repro:
  commit: 1567df1c13e9e805b85f979d11e079c8f5dddc4f
  branch: codex/trt-gvhmr
  pytorch_commit: a845ef9912684fdb527eff8b015560cef4602442
  pytorch_profile_followup_commit: 2f4114a2bbe34e5647a9b1edfed9ec53f74d5dcc
---

# GVHMR SIGA24の可変長5条件比較

RTX 5060 Ti上で、5条件とも145・177フレームの2区分で実測した。TensorRT FP16の推論中央値は2.51 / 2.50 msで、同じ長さのPyTorch FP32に対して7.55 / 6.62倍。カメラ座標のCOCO-17関節差P95は8.61 / 8.72 mm。INT8・FP8ではDRAM readが減り、関節差P95は36.95–94.54 mmだった。

## 固定入力と動的shape

評価は `clip_037` の145フレームと `clip_000` の177フレームで、いずれも選択済み4人物の連続系列。PTQは `clip_027` の117フレーム4系列。評価とPTQでクリップ・動画・同一cameraの収録元frame区間は非重複。全ネットワーク入力、各mask、上流成果物を固定した。 [入力の来歴と一覧](../../runs/run-i2305-gvhmr-tensorrt/README.md)・[tensor lock](../../runs/run-i2305-gvhmr-tensorrt/tensor-lock.json)

profileはB=1とmin/opt/max 117/145/177。3つのエンジンはすべて117・145・177を実行し、116・178をTensorRTの `setInputShape` で拒否して終了値1を返した。各shapeのcontextとInspectorを照合し、FP16は110演算、INT8・FP8は指定精度96演算とFP16 14演算を確認した。 [shape profile](../../runs/run-i2305-gvhmr-tensorrt/shape-profile.json)・[変換・実行の証拠](../../runs/run-i2305-gvhmr-tensorrt/validation.json)

## 速度・メモリ・GPU内部指標

B=1のdenoiserで30回のwarm-upと300回の同期済み測定を行い、入力転送とSMPL-X後処理は区間の外に置いた。メモリはNVMLのデバイス使用量を同期点で採取した最大値。GPU内部指標は別プロセスの固定1系列で採取し、全10組でnsysとncuのkernel数が一致した。stall欄は4種類の時間加重値のうち最大。DRAM writeの0 Bはこの計測区間のカウンタ値。 [手順と統計量](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md)・[全指標とraw成果物の照合先](../../runs/run-i2305-gvhmr-tensorrt/metrics.json)

| L | 条件 | 中央値 / P95 ms | 系列/s / フレーム/s | NVMLピーク MiB | DRAM R/W bytes | Kernel数 | 最大stall | Tensor Core active |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 145 | PyTorch FP32 | 18.942 / 25.603 | 50.9 / 7,382 | 652.1 | 275,522,304 / 0 | 730 | mio_throttle 21.5% | 0.0% |
| 145 | PyTorch FP16 | 14.650 / 16.872 | 67.5 / 9,780 | 558.1 | 106,659,072 / 0 | 701 | long_scoreboard 18.6% | 7.3% |
| 145 | TensorRT FP16 | 2.509 / 3.170 | 393.6 / 57,068 | 524.1 | 71,316,992 / 0 | 135 | long_scoreboard 14.2% | 20.2% |
| 145 | TensorRT INT8 | 2.332 / 3.171 | 411.6 / 59,685 | 494.1 | 39,428,352 / 0 | 150 | long_scoreboard 13.8% | 11.3% |
| 145 | TensorRT FP8 | 2.328 / 3.099 | 411.8 / 59,706 | 492.1 | 40,694,016 / 0 | 151 | long_scoreboard 12.0% | 10.1% |
| 177 | PyTorch FP32 | 16.541 / 18.870 | 59.5 / 10,528 | 654.1 | 345,229,056 / 0 | 867 | mio_throttle 22.5% | 0.0% |
| 177 | PyTorch FP16 | 14.115 / 16.458 | 69.2 / 12,248 | 560.1 | 150,749,696 / 512 | 732 | long_scoreboard 20.2% | 6.4% |
| 177 | TensorRT FP16 | 2.498 / 2.979 | 394.2 / 69,780 | 524.1 | 71,583,232 / 0 | 135 | long_scoreboard 14.2% | 19.3% |
| 177 | TensorRT INT8 | 2.378 / 3.014 | 409.7 / 72,508 | 494.1 | 40,538,624 / 0 | 150 | long_scoreboard 12.6% | 11.0% |
| 177 | TensorRT FP8 | 2.512 / 3.063 | 391.7 / 69,330 | 492.1 | 42,071,040 / 0 | 151 | long_scoreboard 11.6% | 9.9% |

TensorRT FP16ではPyTorch FP32の730 / 867 kernelが135 / 135へ減り、Tensor Core activeは約19–20%だった。速度差には局所attention maskのテンソル演算化と層融合が含まれる。PTQではFP16の約71 MBのreadが約39–42 MBまで減少した。今回の3つのTensorRT条件の中央値は2.33–2.51 msの範囲だった。

## 単体精度

同じFP32の復号・静止カメラ補正・IK・COCO-17再構成を用い、各条件を同じ系列のPyTorch FP32と照合した。各条件でlengthとmaskは全1,288フレーム分が一致し、3つのheadに非有限値はなかった。FP32の復号出力は通常の `Pipeline` を通した結果と一致した。表はP95。座標系を併記した列はカメラ座標 / 重力整列座標の順。 [精度の中央値・P95・最大値と系列別の値](../../runs/run-i2305-gvhmr-tensorrt/metrics.json)

| L | 条件 | body_pose ° | global_orient ° | betas絶対差 | transl L2 mm | COCO-17 L2 mm | 有効フレーム |
|---:|---|---:|---:|---:|---:|---:|---:|
| 145 | PyTorch FP32 | 0.000 | 0.000 / 0.000 | 0.00000 | 0.00 / 0.00 | 0.00 / 0.00 | 580/580 |
| 145 | PyTorch FP16 | 0.062 | 0.041 / 0.394 | 0.00108 | 8.51 / 23.85 | 8.65 / 24.05 | 580/580 |
| 145 | TensorRT FP16 | 0.087 | 0.065 / 0.423 | 0.00161 | 8.49 / 23.24 | 8.61 / 23.50 | 580/580 |
| 145 | TensorRT INT8 | 1.848 | 0.848 / 1.110 | 0.05850 | 92.09 / 84.01 | 94.54 / 84.11 | 580/580 |
| 145 | TensorRT FP8 | 1.606 | 0.612 / 0.651 | 0.03472 | 35.56 / 47.21 | 36.95 / 47.61 | 580/580 |
| 177 | PyTorch FP32 | 0.000 | 0.000 / 0.000 | 0.00000 | 0.00 / 0.00 | 0.00 / 0.00 | 708/708 |
| 177 | PyTorch FP16 | 0.063 | 0.047 / 0.196 | 0.00138 | 8.56 / 20.35 | 8.55 / 20.13 | 708/708 |
| 177 | TensorRT FP16 | 0.085 | 0.069 / 0.181 | 0.00184 | 8.68 / 18.55 | 8.72 / 18.32 | 708/708 |
| 177 | TensorRT INT8 | 2.116 | 1.098 / 1.225 | 0.04046 | 80.19 / 49.99 | 81.96 / 51.85 | 708/708 |
| 177 | TensorRT FP8 | 1.815 | 0.717 / 0.808 | 0.05099 | 67.25 / 71.87 | 68.93 / 71.96 | 708/708 |

PyTorch・TensorRTのFP16は、カメラ座標では暫定許容差の回転P95 1°と関節P95 10 mm以内だった。重力整列座標の関節P95は18.32–24.05 mm。INT8・FP8では両区分とも、body_poseの回転差とカメラ座標の関節差が暫定値を超えた。PTQは#2297のmax・ORT CPU設定を採用した。キャリブレーションは117フレームの4系列で、局所attentionを使う145・177フレームは評価側にある。 [PTQ設定](../../runs/run-i2305-gvhmr-tensorrt/ptq.json)

## E2Eで照合する点

標準pipelineはカメラ座標の身体パラメータを三角測量結果に配置する。単体のカメラ座標差に加え、配置後の位置・向き・関節、人物ごとの有効maskを同じclipで照合する必要がある。denoiserの所要時間から、画像特徴の前処理・身体配置・I/Oを含む総時間は算出できない。既存のsummaryが扱う標準pipelineへの採否は、Phase 1のE2E結果と人間の判断に委ねる。
