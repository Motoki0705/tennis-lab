---
id: run-i2297-clipreid-tensorrt
type: run
task: tennis_scene
sequence: 75
recorded_at: '2026-11-20'
title: CLIP-ReIDのPyTorch・TensorRT 5条件をMeiji固定入力で比較
issue: 2297
provider: codex
date: '2026-11-20'
status: done
config:
  model: clipreid_vitb16_market1501
  batch: 1
  eval_clips:
  - clip_000
  - clip_037
  calibration_clip: clip_027
  calibration_method: max
  run_outcome: complete
metrics:
  torch_fp32:
    latency_median_ms: 5.6061
    latency_p95_ms: 5.8276
    images_per_second: 171.83
    nvml_peak_sampled_bytes: 1645465600
    nsys_gpu_kernels: 188
    ncu_dram_read_bytes: 483787520
    ncu_dram_write_bytes: 0
    ncu_tensor_active_pct_time_weighted: 1.3904
    cosine_p05: 1.0
    id_matches_fp32: 236
    eval_crops: 236
  torch_fp16:
    latency_median_ms: 4.7916
    latency_p95_ms: 15.7483
    images_per_second: 167.74
    nvml_peak_sampled_bytes: 1437847552
    nsys_gpu_kernels: 166
    ncu_dram_read_bytes: 191208960
    ncu_dram_write_bytes: 0
    ncu_tensor_active_pct_time_weighted: 19.1665
    cosine_p05: 0.99999676
    id_matches_fp32: 236
    eval_crops: 236
  trt_fp16:
    latency_median_ms: 1.6843
    latency_p95_ms: 4.2777
    images_per_second: 516.94
    nvml_peak_sampled_bytes: 641830912
    nsys_gpu_kernels: 114
    ncu_dram_read_bytes: 161052160
    ncu_dram_write_bytes: 0
    ncu_tensor_active_pct_time_weighted: 29.8803
    cosine_p05: 0.99998727
    id_matches_fp32: 236
    eval_crops: 236
  trt_int8:
    latency_median_ms: 1.3944
    latency_p95_ms: 2.1698
    images_per_second: 663.25
    nvml_peak_sampled_bytes: 518098944
    nsys_gpu_kernels: 117
    ncu_dram_read_bytes: 85307392
    ncu_dram_write_bytes: 0
    ncu_tensor_active_pct_time_weighted: 22.9842
    cosine_p05: 0.48989643
    id_matches_fp32: 96
    eval_crops: 236
  trt_fp8:
    latency_median_ms: 1.6890
    latency_p95_ms: 2.5923
    images_per_second: 526.77
    nvml_peak_sampled_bytes: 513904640
    nsys_gpu_kernels: 139
    ncu_dram_read_bytes: 76959232
    ncu_dram_write_bytes: 0
    ncu_tensor_active_pct_time_weighted: 18.9889
    cosine_p05: 0.98898266
    id_matches_fp32: 236
    eval_crops: 236
artifacts:
  run_dir: knowledge/runs/run-i2297-clipreid-tensorrt
  output_dir: .worktree-outputs/trt-env-protocol/trt-phase1a
parents:
- run-person-stage-precision-meiji-20261118
relations: []
papers: []
tags:
- clip_reid
- tensorrt
- ptq
summary_impact: none
repro:
  branch: codex/trt-env-protocol
---

# CLIP-ReIDの5条件を固定入力で比較

RTX 5060 Ti（CC 12.0）、WSL2、Windows driver 621.24で全5条件のモデル推論・精度比較・nsysとncuの収集を完了した。WindowsでGPUカウンタを許可した後、同じWSL環境の新規CUDAプロセスでカウンタを取得した。追測には最初の計測と同じ固定入力と既存のTensorRT planを使った。

## 固定入力の実測

評価はMeijiの `clip_000/037` 4カメラの236 crop。PTQは `clip_027` の254 cropのみで実施した。動画digest、同一収録のカメラ別frame区間の両方で重複は0件。B=1、warm-up 30回、同期後の推論300回。精度は同一cropへの対応比較で、IDは別カメラgalleryからの最近傍推定。

| 条件 | 中央値 ms | P95 ms | IQR ms | images/s | NVML GiB（idle差） | Torch peak MiB（allocated / reserved） | nsys kernel | cosine 中央値 / P05 | FP32とのID一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 PyTorch FP32 | 5.606 | 5.828 | 0.115 | 171.8 | 1.532 （0.567） | 369.75 / 416.00 | 188 | 1.000000 / 1.000000 | 236/236 |
| 1 PyTorch FP16 | 4.792 | 15.748 | 1.763 | 167.7 | 1.339 （0.374） | 206.02 / 218.00 | 166 | 0.999998 / 0.999997 | 236/236 |
| 2 TensorRT FP16 | 1.684 | 4.278 | 0.270 | 516.9 | 0.598 （0.343） | 0.19 / 2.00 | 114 | 0.999992 / 0.999987 | 236/236 |
| 3a TensorRT INT8 | 1.394 | 2.170 | 0.231 | 663.3 | 0.483 （0.228） | 0.38 / 2.00 | 117 | 0.548070 / 0.489896 | 96/236 |
| 3b TensorRT FP8 | 1.689 | 2.592 | 0.725 | 526.8 | 0.479 （0.224） | 0.38 / 2.00 | 139 | 0.993240 / 0.988983 | 236/236 |

NVMLはデバイスの同期点でのサンプル最大。Torch欄はPyTorch allocatorの最大値で、TensorRTでは主に入出力バッファ。kernel数は固定の1推論。 [条件別の生データ](../../runs/run-i2297-clipreid-tensorrt/metrics.json)

TensorRTの速度とVRAMは実行contextの `nvtx_verbosity=NONE` を明示して再計測した。同一plan、同一入力で236 cropの出力のSHA-256が各条件で一致した。詳細NVTXを継承していた初回からの差は次のとおり。GPUのidle時点の使用量は3条件とも0.710 GiB低く、NVML絶対peakも同じ量だけ低かった。NVMLのidle差分とTorch allocatorのpeakは各条件で初回と同じだった。計測時点が異なるため、latencyの差には実行環境のばらつきが含まれる。 [旧計測と差分](../../runs/run-i2297-clipreid-tensorrt/metrics.json)・[plan照合](../../runs/run-i2297-clipreid-tensorrt/validation.json)

| 条件 | 初回の中央値 / P95 ms | 再計測と初回の差（中央値 / P95 ms） |
|---|---:|---:|
| 2 TensorRT FP16 | 1.364 / 4.014 | +0.320 / +0.264 |
| 3a TensorRT INT8 | 1.336 / 2.381 | +0.059 / −0.211 |
| 3b TensorRT FP8 | 1.485 / 2.411 | +0.204 / +0.181 |

## GPU内部指標と律速の解釈

代表1推論のkernel数は5条件とも前回のnsysと一致した。stall率とTensor Core activeはkernelのGPU時間で加重平均した。全条件でDRAM writeの実測値は0 B。別途256 MiBのCUDA fill 1回ではDRAM writeを258,687,232 B取得し、同じ転送量をFBPA側のカウンタでも確認した。値は測定したkernel実行中のDRAM転送に限る。 [カウンタ照合](../../runs/run-i2297-clipreid-tensorrt/validation.json)

| 条件 | DRAM read bytes | DRAM write bytes | Long scoreboard % | Short scoreboard % | LG throttle % | MIO throttle % | Tensor Core active % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 PyTorch FP32 | 483,787,520 | 0 | 4.65 | 7.89 | 2.44 | 31.20 | 1.39 |
| 1 PyTorch FP16 | 191,208,960 | 0 | 11.84 | 3.94 | 1.36 | 1.62 | 19.17 |
| 2 TensorRT FP16 | 161,052,160 | 0 | 9.47 | 3.67 | 0.42 | 1.59 | 29.88 |
| 3a TensorRT INT8 | 85,307,392 | 0 | 12.29 | 6.56 | 0.46 | 3.25 | 22.98 |
| 3b TensorRT FP8 | 76,959,232 | 0 | 12.10 | 6.21 | 1.23 | 1.34 | 18.99 |

- PyTorch FP32は選択した4種類のstallのうちMIO throttleが最も大きい（31.20%）。メモリ命令を発行するパイプの待ちが目立つ。TensorRT FP16ではFP32比でDRAM readが66.7%、カーネル数が39.4%減り、Tensor Core activeは1.39%から29.88%に上がった。低精度化とTensorRTの効果が重なる比較で、メモリ転送量・発行回数・Tensor演算利用の三つに差が現れた。
- INT8とFP8はTensorRT FP16からDRAM readをそれぞれ47.0%と52.2%減らした。long scoreboardは9.47%から12%台に上がり、カーネル数は114から117と139に増えた。DRAM読込量の差に対して同期込みの中央値の差は小さい。追加のカーネルとデータ依存待ちが残る挙動が見える。各条件のDRAM帯域飽和の判定には、転送量やstall率に加えて帯域利用率の確認が要る。

## 精度と速度

- TensorRT FP16の再計測中央値は先行計測のPyTorch FP32の約30%で、cosine P05は0.999987、ID変更は0件。代表推論のkernel数は188から114。既存計測の `nsys` 件数を照合に使った `ncu` の再プロファイルでも114件だった。
- 保存済みplanのinspectorで主要演算の対応を確認した。条件2では62実行層でONNXの74演算をFP16、3aでは61実行層で73演算をINT8、3bでは60実行層で72演算をFP8と確認した。3aの末尾projection、3bのpatch Convと末尾projectionはFP16。INT8 PTQでは236件中140件でIDが変わった。FP8のIDは一致し、cosine P05の0.988983は草案0.995を下回った。 [実行精度の定義](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md)・[条件別の照合値](../../runs/run-i2297-clipreid-tensorrt/metrics.json)
- INT8の精度差を同じ12 cropで切り分けた。ONNX Runtime CPUでFP32 exportのcosine中央値はPyTorch基準とほぼ1.0、INT8 Q/DQ graphは0.560。量子化graphの時点で劣化が現れ、同じINT8 graphでORT CPUとTensorRTのcosine中央値は0.664だった。差が生じる層や方式ごとの寄与は未特定。 [CPUでの照合結果](../../runs/run-i2297-clipreid-tensorrt/validation.json)
- PyTorch FP16は中央値が短く、P95は長かった。別プロセスで再度測ったFP32の中央値/P95は5.646/5.917 ms、FP16は3.980/13.833 ms。images/sは初回171.8/167.7、再測171.2/193.6で順位が変わった。TensorRT間の差も単一ホストでの観測値として扱う。

## 環境と追測の照合

[専用環境](../../runs/run-i2297-clipreid-tensorrt/README.md) でTensorRT Python importと `trtexec` を確認した。小さなQ/DQ行列積はCC 12.0上でINT8、FP8とも期待値64に一致した。既設nsys 2025.3.2ではGPU kernel activityが欠け、2026.5.2の明示的なsoftware CUDA traceで1件の行列積と全5条件のモデルkernelを取得した。FP16エンジン初回はinspectorの `Half` 表記で拒否された。判定を修正し、再ビルドと計測を完了した。無効精度指定と壊れたONNXは非ゼロで終了し、エンジンが生成されないことを確かめた。 [環境出力](../../runs/run-i2297-clipreid-tensorrt/environment.json)・[検証結果](../../runs/run-i2297-clipreid-tensorrt/validation.json)

初回の `ERR_NVGPUCTRPERM` と、Windows側の設定後に行列積1回で同じ指標を取得した結果を記録した。TensorRTでは詳細NVTXラベル内のJSONで `ncu` のraw CSVの列がずれた。列数を照合する読み取りが停止し、`ncu` の実行contextで詳細layerラベルを抑えて再実行した。各条件のCSV列数・単位・加重集計を独立に照合し、代表入力とplanの識別も確認した。追測時に異なるkernel数を与える回帰テストでは、`coverage_mismatch` と終了値3を確認した。 [検証記録](../../runs/run-i2297-clipreid-tensorrt/validation.json)・[比較プロトコル](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md)

現行pipelineの採否判断は維持する。今回測ったモデルはCLIP-ReID。ViTPose-H・HMR2・DINO・GVHMRの前処理採取位置、可変shapeと許容差草案を [プロトコル](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md) に記録した。
