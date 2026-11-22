---
id: run-i2304-dino-tensorrt
type: run
task: tennis_scene
sequence: 78
recorded_at: '2026-11-21'
title: DINO 4-scale Swin-Lの5精度条件とMSDAのTensorRT実装を比較
issue: 2304
provider: codex
date: '2026-11-20'
status: done
config:
  checkpoint: dino/checkpoint0029_4scale_swin.pth
  dataset: meiji_4cam/video_000
  eval_clips:
  - clip_000
  - clip_037
  calibration_clip: clip_027
  eval_images: 64
  calibration_images: 64
  input_shape_nchw:
  - 1
  - 3
  - 750
  - 1333
  original_image_hw:
  - 1080
  - 1920
  tensor_lock_sha256: 481cf905f03d63ce93d9ca34df8b38093ef661be7776b355daefd9b6f58da96e
  device: NVIDIA GeForce RTX 5060 Ti
  driver: '621.24'
  pytorch: 2.13.0+cu130
  tensorrt: 10.14.1.48
  modelopt: 0.48.0
  precision_protocol_issue: 2297
  calibration_method: max
  calibration_eps:
  - cpu
  confidence_threshold: 0.3
  matching_min_iou: 0.5
  nsight_compute_replay: kernel
metrics:
  pytorch_fp32:
    median_ms: 268.147242
    p95_ms: 272.500553
    iqr_ms: 2.764974
    mad_ms: 1.31761
    images_per_second: 3.721714
    nvml_device_peak_sampled_bytes: 2699137024
    nvml_device_peak_minus_idle_bytes: 2425458688
    torch_peak_allocated_bytes: 1868561408
    nsys_gpu_kernels: 1862
    nsys_sum_kernel_duration_ns: 246242410
    ncu_status: measured
    iou_p05: 1.0
    xyxy_abs_error_px_p95: 0.0
    score_abs_error_p95: 0.0
    matched_boxes: 494
    reference_boxes: 494
    candidate_boxes: 494
    missing_boxes: 0
    extra_boxes: 0
    dram_read_bytes: 26359204864
    dram_write_bytes: 12496338176
    tensor_active_pct_time_weighted: 0.0
    stall_pct_time_weighted:
      long_scoreboard: 28.358453
      short_scoreboard: 5.782284
      lg_throttle: 0.742474
      mio_throttle: 11.32643
  pytorch_fp16:
    median_ms: 154.086959
    p95_ms: 169.25288
    iqr_ms: 4.84597
    mad_ms: 2.052912
    images_per_second: 6.408243
    nvml_device_peak_sampled_bytes: 2705428480
    nvml_device_peak_minus_idle_bytes: 2431750144
    torch_peak_allocated_bytes: 1767208448
    nsys_gpu_kernels: 2370
    nsys_sum_kernel_duration_ns: 126247465
    ncu_status: measured
    iou_p05: 0.996306
    xyxy_abs_error_px_p95: 0.03707
    score_abs_error_p95: 0.006019
    matched_boxes: 493
    reference_boxes: 494
    candidate_boxes: 493
    missing_boxes: 1
    extra_boxes: 0
    dram_read_bytes: 21208379136
    dram_write_bytes: 8193342208
    tensor_active_pct_time_weighted: 14.583001
    stall_pct_time_weighted:
      long_scoreboard: 45.641467
      short_scoreboard: 4.536334
      lg_throttle: 0.579338
      mio_throttle: 3.138546
  tensorrt_fp16_gridsample:
    median_ms: 56.980689
    p95_ms: 57.324101
    iqr_ms: 0.332982
    mad_ms: 0.163849
    images_per_second: 17.543746
    nvml_device_peak_sampled_bytes: 1203867648
    nvml_device_peak_minus_idle_bytes: 930189312
    torch_peak_allocated_bytes: 13498880
    nsys_gpu_kernels: 561
    nsys_sum_kernel_duration_ns: 54847609
    ncu_status: measured
    iou_p05: 0.93252
    xyxy_abs_error_px_p95: 0.707361
    score_abs_error_p95: 0.017263
    matched_boxes: 491
    reference_boxes: 494
    candidate_boxes: 493
    missing_boxes: 3
    extra_boxes: 2
    dram_read_bytes: 6478485760
    dram_write_bytes: 2619615232
    tensor_active_pct_time_weighted: 26.203678
    stall_pct_time_weighted:
      long_scoreboard: 52.693116
      short_scoreboard: 4.466582
      lg_throttle: 0.470955
      mio_throttle: 2.791505
  tensorrt_int8_gridsample:
    median_ms: 45.13551
    p95_ms: 45.506606
    iqr_ms: 0.265529
    mad_ms: 0.128433
    images_per_second: 22.135971
    nvml_device_peak_sampled_bytes: 1073844224
    nvml_device_peak_minus_idle_bytes: 800165888
    torch_peak_allocated_bytes: 13498880
    nsys_gpu_kernels: 585
    nsys_sum_kernel_duration_ns: 43133043
    ncu_status: unavailable
    iou_p05: 0.925335
    xyxy_abs_error_px_p95: 0.910519
    score_abs_error_p95: 0.051673
    matched_boxes: 483
    reference_boxes: 494
    candidate_boxes: 488
    missing_boxes: 11
    extra_boxes: 5
  tensorrt_fp8_gridsample:
    median_ms: 52.554699
    p95_ms: 52.840223
    iqr_ms: 0.208895
    mad_ms: 0.104078
    images_per_second: 19.020667
    nvml_device_peak_sampled_bytes: 971083776
    nvml_device_peak_minus_idle_bytes: 697405440
    torch_peak_allocated_bytes: 13498880
    nsys_gpu_kernels: 608
    nsys_sum_kernel_duration_ns: 50748249
    ncu_status: measured
    iou_p05: 0.913169
    xyxy_abs_error_px_p95: 0.917671
    score_abs_error_p95: 0.066248
    matched_boxes: 489
    reference_boxes: 494
    candidate_boxes: 500
    missing_boxes: 5
    extra_boxes: 11
    dram_read_bytes: 5367527424
    dram_write_bytes: 2215776512
    tensor_active_pct_time_weighted: 16.735272
    stall_pct_time_weighted:
      long_scoreboard: 44.594766
      short_scoreboard: 5.157713
      lg_throttle: 0.568202
      mio_throttle: 2.10626
  tensorrt_fp16_plugin_comparison:
    median_ms: 41.474508
    p95_ms: 41.8178
    iqr_ms: 0.215826
    mad_ms: 0.104779
    images_per_second: 24.094106
    nvml_device_peak_sampled_bytes: 1140953088
    nvml_device_peak_minus_idle_bytes: 867274752
    torch_peak_allocated_bytes: 13498880
    nsys_gpu_kernels: 560
    nsys_sum_kernel_duration_ns: 39670719
    ncu_status: out_of_scope
    iou_p05: 0.930956
    xyxy_abs_error_px_p95: 0.710608
    score_abs_error_p95: 0.019537
    matched_boxes: 490
    reference_boxes: 494
    candidate_boxes: 494
    missing_boxes: 4
    extra_boxes: 4
artifacts:
  run_dir: knowledge/runs/run-i2304-dino-tensorrt
  output_dir: .worktree-outputs/trt-dino/trt-dino
parents:
- run-i2297-clipreid-tensorrt
- run-person-stage-precision-meiji-20261118
relations: []
papers: []
tags:
- dino
- tensorrt
- inference
- ptq
summary_impact: none
repro:
  branch: codex/trt-dino
  remote: git@github.com:Motoki0705/tennis-lab.git
---

## 同一入力での比較

RTX 5060 Tiで評価64枚と校正64枚を固定した。評価は `clip_000`・`clip_037`、校正は `clip_027`。全画像を現行DINOの前処理で変換し、B1、H×W 750×1333の1 bucketで比較した。TensorRT planのmin・opt・maxは同一で、元画像H/Wは1080×1920。評価と校正は動画ファイル・元フレーム区間とも共通件数0。 [入力lock](../../runs/run-i2304-dino-tensorrt/tensor-lock.json)

時間は30回のwarm-up後、同期込みの3×100回。GPU上の画像入力からモデルの出力テンソルまでを測った。VRAMはNVMLで採取したデバイス全体のpeak。box照合は元画像座標へのdecode後、confidence 0.30とIoU 0.50で行い、対応成立ペアの精度を算出した。代表入力1枚からGPU内部指標を採取し、stallとTensor Core activeはkernel時間で加重した。 [全カウンタ、精度分布、clip・camera別結果](../../runs/run-i2304-dino-tensorrt/metrics.json)

| 条件 | 中央値 / P95 ms | 枚/s | Peak GiB | kernels | IoU P05 | xyxy絶対誤差 P95 px | score絶対誤差 P95 | 欠落 / 追加 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 PyTorch FP32 | 268.15 / 272.50 | 3.72 | 2.51 | 1,862 | 1.0000 | 0 | 0 | 0 / 0 |
| 1 PyTorch FP16 | 154.09 / 169.25 | 6.41 | 2.52 | 2,370 | 0.9963 | 0.0371 | 0.0060 | 1 / 0 |
| 2 TRT FP16 GridSample | 56.98 / 57.32 | 17.54 | 1.12 | 561 | 0.9325 | 0.7074 | 0.0173 | 3 / 2 |
| 3a TRT INT8 GridSample | 45.14 / 45.51 | 22.14 | 1.00 | 585 | 0.9253 | 0.9105 | 0.0517 | 11 / 5 |
| 3b TRT FP8 GridSample | 52.55 / 52.84 | 19.02 | 0.90 | 608 | 0.9132 | 0.9177 | 0.0662 | 5 / 11 |
| 比較 TRT FP16プラグイン | 41.47 / 41.82 | 24.09 | 1.06 | 560 | 0.9310 | 0.7106 | 0.0195 | 4 / 4 |

条件1は #2238 と同じPyTorch CUDA FP16 autocastとFP32 MSDAブロックを使った。基準の494 boxに対して1件が欠落し、単体許容差の草案を満たした。正式な条件2はFP32比で中央値4.71倍の速度で、代表入力のDRAM readは26.36 GBから6.48 GB、kernelは1,862から561に減った。TRTの3条件と比較用プラグインは、いずれもIoU P05とscore差P95が草案の範囲を外れた。 [#2238のrun](000050-run-person-stage-precision-meiji-20261118.md)・[精度許容差](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md#精度の指標と許容差草案)

条件2の精度低下の原因は未確認。条件1のMSDAブロック全体はFP32で、条件2はMSDA方式を変え、Swinなどを含むONNX全体をFP16化してTensorRTで実行した。固定入力でのモデル全体の差を観測したが、方式・変換・精度配分の寄与は未分離。同一精度でのexport前後の出力照合と、モデル全体でMSDAだけをFP32に戻す対照は未実施。 [精度比較の内訳](../../runs/run-i2304-dino-tensorrt/metrics.json)

## MSDAと量子化

小規模比較ではGridSample、TensorRT v2プラグイン、FP16射影とPyTorch FP32 CUDA MSDAを組み合わせた部分TRT化が動作した。モデル全体の条件2ではプラグインの中央値がGridSampleより27.2%短く、GPU kernel時間合計は54.85 msと39.67 msだった。box一致精度は上表の範囲で変化した。GridSampleの標準ONNXはCPU校正にも使用できた。プラグインはTensorRT配布版のライブラリとplanの版管理を要し、配布にない環境ではビルドも必要になる。プラグイン版のPTQと部分TRT化のモデル全体は未計測。 [3方式の速度・誤差・運用負担](../../runs/run-i2304-dino-tensorrt/README.md#msdaの方式比較)

DINOの3a・3bは計測時点の #2297 のModelOpt max、ORT CPU、明示Q/DQで測った。主要演算のオペランドは3aがINT8 292・FP16 24、3bがFP8 E4M3 275・FP16 41だった。実測ONNXでは指定外のAdd 36件、LayerNormalization 5件、Where 1件が両条件でDQ出力を直接受けた。3a・3bの結果は、この指定外の数値入力まで量子化が及んだ実測グラフでの値。現行の変換はDQの接続先を照合し、旧グラフを拒否する。#2334のViT向け設定でのDINOは未計測で、再計測は #2278 で判断する。 [演算ごとの精度](../../runs/run-i2304-dino-tensorrt/precision-requirements.json)・[実測グラフのDQ接続](../../runs/run-i2304-dino-tensorrt/metrics.json)・[ViT 3モデルのPTQ設定](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md#vit-3モデルのptq設定)

## 採取範囲と下流への影響

3aのDRAM R/W、stall、Tensor Core activeは正式結果では未採取。v6のkernel replayは45分、同一planでのv7は12分の上限に達し、判定時点でrawのkernel行が0件だった。v6の遅れて書き出されたCSVとv7のapplication replay診断は試行記録に残した。v7の診断は585カーネルでnsysと一致し、jobはkernel replayの失敗を示す終了値3を返した。条件2のプラグイン版は合意済みの速度・VRAM・単体精度・nsysまでを収集した。 [試行・終了値と診断値](../../runs/run-i2304-dino-tensorrt/attempts.json)

実行層の照合に失敗した変換と、動作範囲外の749×1333入力は非ゼロで終了した。部分TRT化はTensorRT FP16射影からFP32 castを経てPyTorch CUDA MSDAへ渡す境界を記録した。 [非ゼロ終了の抜粋](../../runs/run-i2304-dino-tensorrt/attempts.json)・[部分TRTの境界と入出力精度](../../runs/run-i2304-dino-tensorrt/comparison.json)

現行の本番構成は`summary.md` のPyTorch FP16を使う。今回のB1・1 shapeでの単体結果から本番構成の判断を更新せず、MSDAの採否は #2278 に返す。E2Eでは新しいboxを入力した追跡ID、カメラ間の人物割当、3Dの欠測と誤差、および横断batch時のスループットを確認する余地が残る。 [現行構成](../../summary.md#標準pipelinetennis_scene)
