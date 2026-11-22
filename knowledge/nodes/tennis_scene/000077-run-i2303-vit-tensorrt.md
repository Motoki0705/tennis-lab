---
id: run-i2303-vit-tensorrt
type: run
task: tennis_scene
sequence: 77
recorded_at: '2026-11-20'
title: ViTPose-H・HMR2の5精度条件とViT系PTQの比較
issue: 2303
provider: codex
date: '2026-11-20'
status: done
config:
  device: NVIDIA GeForce RTX 5060 Ti
  driver: '621.24'
  dataset: meiji_4cam/video_000
  eval_clips:
  - clip_000
  - clip_037
  calibration_clip: clip_027
  model_checkpoints:
    vitpose_h: vitpose/vitpose-h-multi-coco.pth
    hmr2: hmr2/epoch=10-step=25000.ckpt
  input_batch_size: 1
  vitpose_flip_included: true
  tensor_lock_sha256: 757984765eb0a9f069154dc3e143121b9ede97f6223806e10cef971a081a4b8e
  ptq_int8: attention-out-max
  ptq_fp8: linear-max
metrics:
  vitpose_h:
    pytorch_fp32:
      median_ms: 54.42296
      p95_ms: 55.146595
      images_per_second: 18.3118
      torch_peak_allocated_bytes: 2673240576
      nvml_device_peak_sampled_bytes: 3196162048
      dram_read_bytes: 3966944768
      dram_write_bytes: 145381632
      gpu_kernels: 634
      xy_l2_px_p95: 0.0
      xy_l2_px_max: 0.0
      confidence_abs_p95: 0.0
    pytorch_fp16:
      median_ms: 27.196799
      p95_ms: 28.322378
      images_per_second: 36.5749
      torch_peak_allocated_bytes: 2677986304
      nvml_device_peak_sampled_bytes: 3198259200
      dram_read_bytes: 4701833472
      dram_write_bytes: 40244224
      gpu_kernels: 994
      xy_l2_px_p95: 0.007683
      xy_l2_px_max: 0.104256
      confidence_abs_p95: 0.000618
    tensorrt_fp16:
      median_ms: 8.467616
      p95_ms: 8.608277
      images_per_second: 117.36
      torch_peak_allocated_bytes: 1492992
      nvml_device_peak_sampled_bytes: 1763807232
      dram_read_bytes: 1478578688
      dram_write_bytes: 6400
      gpu_kernels: 238
      xy_l2_px_p95: 0.02471
      xy_l2_px_max: 0.30197
      confidence_abs_p95: 0.001592
    tensorrt_int8:
      median_ms: 8.690997
      p95_ms: 8.799465
      images_per_second: 114.8764
      torch_peak_allocated_bytes: 2985984
      nvml_device_peak_sampled_bytes: 1721864192
      dram_read_bytes: 1368989184
      dram_write_bytes: 0
      gpu_kernels: 300
      xy_l2_px_p95: 0.079412
      xy_l2_px_max: 23.353573
      confidence_abs_p95: 0.005194
    tensorrt_fp8:
      median_ms: 7.731699
      p95_ms: 8.075113
      images_per_second: 125.62
      torch_peak_allocated_bytes: 2985984
      nvml_device_peak_sampled_bytes: 1145147392
      dram_read_bytes: 767607040
      dram_write_bytes: 0
      gpu_kernels: 270
      xy_l2_px_p95: 0.283881
      xy_l2_px_max: 29.059164
      confidence_abs_p95: 0.019175
  hmr2:
    pytorch_fp32:
      median_ms: 36.563522
      p95_ms: 39.089333
      images_per_second: 27.0626
      torch_peak_allocated_bytes: 2799349760
      nvml_device_peak_sampled_bytes: 3326185472
      dram_read_bytes: 3376803584
      dram_write_bytes: 22684928
      gpu_kernels: 705
      cosine_p05: 1.0
      relative_l2_p95: 0.0
    pytorch_fp16:
      median_ms: 25.058203
      p95_ms: 28.534885
      images_per_second: 39.0588
      torch_peak_allocated_bytes: 2807058432
      nvml_device_peak_sampled_bytes: 3326185472
      dram_read_bytes: 4487879936
      dram_write_bytes: 39337728
      gpu_kernels: 1131
      cosine_p05: 0.99999838
      relative_l2_p95: 0.001799766
    tensorrt_fp16:
      median_ms: 6.537979
      p95_ms: 6.960165
      images_per_second: 150.8555
      torch_peak_allocated_bytes: 399360
      nvml_device_peak_sampled_bytes: 1793167360
      dram_read_bytes: 1336494848
      dram_write_bytes: 0
      gpu_kernels: 309
      cosine_p05: 0.999991978
      relative_l2_p95: 0.004005616
    tensorrt_int8:
      median_ms: 6.573377
      p95_ms: 6.965741
      images_per_second: 150.7308
      torch_peak_allocated_bytes: 790528
      nvml_device_peak_sampled_bytes: 1761710080
      dram_read_bytes: 1286221568
      dram_write_bytes: 0
      gpu_kernels: 341
      cosine_p05: 0.99990905
      relative_l2_p95: 0.013965331
    tensorrt_fp8:
      median_ms: 5.399121
      p95_ms: 5.953242
      images_per_second: 182.0895
      torch_peak_allocated_bytes: 790528
      nvml_device_peak_sampled_bytes: 1187090432
      dram_read_bytes: 709049600
      dram_write_bytes: 0
      gpu_kernels: 341
      cosine_p05: 0.99965411
      relative_l2_p95: 0.026228653
  clip:
    tensorrt_int8:
      median_ms: 1.574794
      p95_ms: 2.331818
      images_per_second: 599.2885
      torch_peak_allocated_bytes: 398336
      nvml_device_peak_sampled_bytes: 612470784
      dram_read_bytes: 155445248
      dram_write_bytes: 0
      gpu_kernels: 127
      accuracy_crops: 224
      cosine_p05: 0.999601983
      id_matches_fp32: 224
    tensorrt_fp8:
      median_ms: 1.664813
      p95_ms: 2.277651
      images_per_second: 573.2404
      torch_peak_allocated_bytes: 398336
      nvml_device_peak_sampled_bytes: 513904640
      dram_read_bytes: 75960832
      dram_write_bytes: 0
      gpu_kernels: 115
      accuracy_crops: 224
      cosine_p05: 0.995416171
      id_matches_fp32: 224
artifacts:
  run_dir: knowledge/runs/run-i2303-vit-tensorrt
  output_dir: .worktree-outputs/trt-vit-pose-hmr2/i2303
  metrics: knowledge/runs/run-i2303-vit-tensorrt/metrics.json
  tensor_lock: knowledge/runs/run-i2303-vit-tensorrt/tensor-lock.json
  int8_diagnosis: knowledge/runs/run-i2303-vit-tensorrt/int8-diagnosis.json
  validation: knowledge/runs/run-i2303-vit-tensorrt/validation.json
parents:
- run-i2297-clipreid-tensorrt
relations: []
papers: []
tags:
- tensorrt
- ptq
- vitpose
- hmr2
- clip_reid
summary_impact: updated
repro:
  branch: codex/trt-vit-pose-hmr2
  readme: knowledge/runs/run-i2303-vit-tensorrt/README.md
---

# ViTPose-H・HMR2の5精度条件とViT系PTQの比較

ViTPose-HではTensorRT FP16が座標とconfidenceの許容差を満たし、中央値は54.423 msから8.468 msになった。INT8・FP8ではそれぞれ1点・6点で座標の最大差5 pxを超えた。HMR2では5条件とも単体精度の判定を満たし、FP8の中央値が5.399 msと最短だった。CLIP-ReIDの新INT8・FP8は、選定用cropを除いた224件でcosineとID一致の許容差を満たした。

## 固定入力と計測区間

RTX 5060 Ti（CC 12.0）、WSL2、driver 621.24、PyTorch 2.13.0+cu130、TensorRT 10.14.1.48.post1、ModelOpt 0.48.0。評価は `clip_000`・`clip_037`、PTQ校正は `clip_027`。媒体digestの重複も、同じ収録元のカメラ別フレーム区間の重複も0で、独立に再展開したテンソルはlockと一致した。 [固定入力](../../runs/run-i2303-vit-tensorrt/tensor-lock.json)・[照合記録](../../runs/run-i2303-vit-tensorrt/validation.json)

| モデル | 評価 / 校正crop | 入力と区間 |
|---|---:|---|
| ViTPose-H | 236 / 254 | B=1、内部は反転込み2×3×256×192。GPU上の反転と連結、forward、ヒートマップの反転復元と左右交換・平均、CUDA同期まで |
| HMR2 | 40 / 32 | 1×3×256×256。body viewで選ばれたカメラ・実フレーム・boxを使用。モデル内の幅切り出しから1024次元の画像特徴とCUDA同期まで |
| CLIP-ReID（追測） | 236 / 254 | #2297と同じ1×3×256×128、visual towerとCLS・projectionの連結まで |

条件ごとに独立したプロセスで30回warm-upし、300回の同期込み推論を100回ずつ3ブロックで計測した。条件0はTF32無効のFP32。ViTPose-HとHMR2の条件1はFP16 autocast。外部のcrop生成・入出力転送・UDP復元は時刻計測の外で行った。 [再現手順](../../runs/run-i2303-vit-tensorrt/README.md)・[比較プロトコル](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md)

## 速度とPeak VRAM

throughputは同期済み時間の総和から求めた。NVML欄は同期点でのデバイス全体の最大使用量とidleからの差。Torch allocatorのうちTensorRT条件で観測できるのは主に入出力バッファ。 [ブロック別の時刻とメモリ値](../../runs/run-i2303-vit-tensorrt/metrics.json)

| モデル | 条件 | 中央値 / P95 (ms) | IQR / MAD (ms) | crop/s | Torch allocated / reserved (MiB) | NVML peak / idle差 (MiB) |
|---|---|---:|---:|---:|---:|---:|
| ViTPose-H | 0 PyTorch FP32 | 54.423 / 55.147 | 0.338 / 0.127 | 18.31 | 2,549.40 / 2,622.00 | 3,048.1 / 2,787.1 |
| ViTPose-H | 1 PyTorch FP16 | 27.197 / 28.322 | 0.624 / 0.312 | 36.57 | 2,553.93 / 2,624.00 | 3,050.1 / 2,789.1 |
| ViTPose-H | 2 TensorRT FP16 | 8.468 / 8.608 | 0.089 / 0.044 | 117.36 | 1.42 / 2.00 | 1,682.1 / 1,421.1 |
| ViTPose-H | 3a TensorRT INT8 | 8.691 / 8.799 | 0.066 / 0.032 | 114.88 | 2.85 / 24.00 | 1,642.1 / 1,381.1 |
| ViTPose-H | 3b TensorRT FP8 | 7.732 / 8.075 | 0.097 / 0.049 | 125.62 | 2.85 / 24.00 | 1,092.1 / 831.1 |
| HMR2 | 0 PyTorch FP32 | 36.564 / 39.089 | 1.051 / 0.434 | 27.06 | 2,669.67 / 2,746.00 | 3,172.1 / 2,911.1 |
| HMR2 | 1 PyTorch FP16 | 25.058 / 28.535 | 1.392 / 0.620 | 39.06 | 2,677.02 / 2,746.00 | 3,172.1 / 2,911.1 |
| HMR2 | 2 TensorRT FP16 | 6.538 / 6.960 | 0.212 / 0.107 | 150.86 | 0.38 / 2.00 | 1,710.1 / 1,449.1 |
| HMR2 | 3a TensorRT INT8 | 6.573 / 6.966 | 0.201 / 0.101 | 150.73 | 0.75 / 2.00 | 1,680.1 / 1,419.1 |
| HMR2 | 3b TensorRT FP8 | 5.399 / 5.953 | 0.376 / 0.141 | 182.09 | 0.75 / 2.00 | 1,132.1 / 871.1 |
| CLIP-ReID | 3a TensorRT INT8 | 1.575 / 2.332 | 0.193 / 0.086 | 599.29 | 0.38 / 2.00 | 584.1 / 323.1 |
| CLIP-ReID | 3b TensorRT FP8 | 1.665 / 2.278 | 0.430 / 0.206 | 573.24 | 0.38 / 2.00 | 490.1 / 229.1 |

## GPU内部指標

代表cropは `clip_000/cam0/frame 178/person 0`。Nsight SystemsとNsight Computeのカーネル件数は全12条件で一致した。stallは取得した4指標のうち最大の理由を掲載する。stall率とTensor Core activeはカーネルのGPU時間で加重した。 [全stallとカーネル内訳](../../runs/run-i2303-vit-tensorrt/metrics.json)

| モデル | 条件 | DRAM read (B) | DRAM write (B) | kernel数 | 最大のstall | Tensor Core active |
|---|---|---:|---:|---:|---:|---:|
| ViTPose-H | 0 PyTorch FP32 | 3,966,944,768 | 145,381,632 | 634 | mio_throttle 18.44% | 0.00% |
| ViTPose-H | 1 PyTorch FP16 | 4,701,833,472 | 40,244,224 | 994 | long_scoreboard 40.72% | 22.18% |
| ViTPose-H | 2 TensorRT FP16 | 1,478,578,688 | 6,400 | 238 | long_scoreboard 4.14% | 53.72% |
| ViTPose-H | 3a TensorRT INT8 | 1,368,989,184 | 0 | 300 | long_scoreboard 2.83% | 29.53% |
| ViTPose-H | 3b TensorRT FP8 | 767,607,040 | 0 | 270 | long_scoreboard 3.70% | 35.13% |
| HMR2 | 0 PyTorch FP32 | 3,376,803,584 | 22,684,928 | 705 | mio_throttle 15.74% | 0.00% |
| HMR2 | 1 PyTorch FP16 | 4,487,879,936 | 39,337,728 | 1131 | long_scoreboard 52.27% | 15.64% |
| HMR2 | 2 TensorRT FP16 | 1,336,494,848 | 0 | 309 | long_scoreboard 4.88% | 31.05% |
| HMR2 | 3a TensorRT INT8 | 1,286,221,568 | 0 | 341 | long_scoreboard 4.10% | 24.44% |
| HMR2 | 3b TensorRT FP8 | 709,049,600 | 0 | 341 | long_scoreboard 5.58% | 26.24% |
| CLIP-ReID | 3a TensorRT INT8 | 155,445,248 | 0 | 127 | long_scoreboard 9.11% | 28.93% |
| CLIP-ReID | 3b TensorRT FP8 | 75,960,832 | 0 | 115 | long_scoreboard 16.64% | 17.11% |

両モデルのTensorRT FP16はFP32よりDRAM readとカーネル数が減少した。INT8ではFP16から読込量がさらに減り、カーネル数はViTPose-Hで238から300、HMR2で309から341に増えた。同期込みの中央値は両モデルでFP16と近く、FP8ではHMR2の読込量がFP16比で47%減り、中央値は6.538 msから5.399 msになった。転送量とstall率だけで帯域の飽和は判定できない。

## 単体精度

同じ固定入力のPyTorch FP32出力を基準にした。ViTPose-Hの基準はxy距離のP95が1 px以下、最大が5 px以下、confidence絶対差のP95が0.02以下。元画像17関節の有効マスクは全条件で一致し、4,012点を比較できた。INT8で5 pxを超えたのは `clip_000/cam3/frame 260/person 3` の鼻で、ヒートマップ最大位置が (30,0) から (22,13) に移った。FP8で5 pxを超えた6点は4 cropに分布した。 [外れ値のアドレスと値](../../runs/run-i2303-vit-tensorrt/metrics.json)

| ViTPose-H | xy L2 P95 / 最大 (px) | confidence絶対差 P95 | mask不一致 | 許容差 |
|---|---:|---:|---:|---|
| 0 PyTorch FP32 | 0.00000 / 0.00000 | 0.00000 | 0 | 適合 |
| 1 PyTorch FP16 | 0.00768 / 0.10426 | 0.00062 | 0 | 適合 |
| 2 TensorRT FP16 | 0.02471 / 0.30197 | 0.00159 | 0 | 適合 |
| 3a TensorRT INT8 | 0.07941 / 23.35357 | 0.00519 | 0 | 最大差超過 |
| 3b TensorRT FP8 | 0.28388 / 29.05916 | 0.01918 | 0 | 最大差超過 |

HMR2では対応フレーム・欠測のずれは0件。基準はcosine P05が0.995以上、相対L2 P95が0.05以下。FP8の相対L2最大値は0.076740で、次段への影響はE2Eで確認する。 [フレームごとの相対差と絶対差の分布](../../runs/run-i2303-vit-tensorrt/metrics.json)・[次元別の最大絶対差](../../runs/run-i2303-vit-tensorrt/hmr2-feature-errors.json)

| HMR2 | cosine P05 | 相対L2 P95 / 最大 | 許容差 |
|---|---:|---:|---|
| 0 PyTorch FP32 | 1.00000000 | 0.000000 / 0.000000 | 適合 |
| 1 PyTorch FP16 | 0.99999838 | 0.001800 / 0.002274 | 適合 |
| 2 TensorRT FP16 | 0.99999198 | 0.004006 / 0.006143 | 適合 |
| 3a TensorRT INT8 | 0.99990905 | 0.013965 / 0.018077 | 適合 |
| 3b TensorRT FP8 | 0.99965411 | 0.026229 / 0.076740 | 適合 |

CLIP-ReIDはPTQ設定の選定に使った12 cropをqueryとgalleryの両方から外し、同じ2クリップ内の残り224件で精度を判定した。保存済みの特徴から条件ごとにgalleryを作り直した。両条件ともcosine P05 ≥ 0.995、FP32とのID一致 ≥ 99%を満たした。旧条件との比較用に全236件の集計も残す。 [除外した入力と両方の集計](../../runs/run-i2303-vit-tensorrt/metrics.json)

| CLIP-ReID | 集計の用途 | cosine P05 | FP32とのID一致 |
|---|---|---:|---:|
| 3a TensorRT INT8 | 精度判定 | 0.999602 | 224/224 |
| 3b TensorRT FP8 | 精度判定 | 0.995416 | 224/224 |
| #2297の旧INT8 | 同一入力比較 | 0.489896 | 96/236 |
| #2297の旧FP8 | 同一入力比較 | 0.988983 | 236/236 |
| 3a TensorRT INT8 | 同一入力比較 | 0.999602 | 236/236 |
| 3b TensorRT FP8 | 同一入力比較 | 0.995429 | 236/236 |

## CLIP-ReIDのINT8劣化の切り分け

旧Q/DQ ONNXを同じ12 cropで比較し、量子化された入力辺を種別・ブロックごとに元の辺へ戻した。活性化を元に戻すとcosine P05が0.51539から0.99974に回復し、重み・定数だけを戻した場合は0.48647だった。層種ではLayerNorm入力が大きく寄与し、LayerNorm・QuickGELU・残差Addを合わせると0.84776まで回復した。単一ブロックの活性化を戻した場合の最高値はblock 7の0.54358だった。softmaxには旧グラフの直接Q/DQ入力辺がなかった。 [辺単位・層種・ブロック別の完全な比較](../../runs/run-i2303-vit-tensorrt/int8-diagnosis.json)

| 旧グラフから戻した入力辺 | 辺数 | cosine P05 |
|---|---:|---:|
| なし | 0 | 0.51539 |
| 活性化すべて | 181 | 0.99974 |
| 重み・定数すべて | 49 | 0.48647 |
| LayerNormの活性化 | 24 | 0.74920 |
| QuickGELUの活性化 | 36 | 0.54217 |
| 残差Addの活性化 | 24 | 0.52286 |
| LayerNorm・QuickGELU・残差Addの活性化 | 84 | 0.84776 |
| attention MatMulの可変入力 | 48 | 0.51127 |

同じ32校正cropの比較で、対象を重み付き線形層に限定するとP05は0.84918、同じ層のentropy校正では0.85612だった。また、maxを固定して投影ごとにQ/DQを入れると、attention出力だけでは0.99971、MLP出力だけでは0.86000だった。旧設定の劣化は、校正方式を切り替えるより活性化の量子化範囲を絞る方が大きく改善した。固定版ModelOptの通常のINT8経路はmaxとentropyを使い分けるため、校正方式の比較はこの2種とした。

| Q/DQ対象と校正 | cosine P05 |
|---|---:|
| デフォルト対象 / max | 0.52635 |
| MatMul・Gemm / max | 0.85032 |
| 重み付き線形層 / max | 0.84918 |
| 重み付き線形層 / entropy | 0.85612 |
| QKV投影のみ / max | 0.99727 |
| attention出力投影のみ / max | 0.99971 |
| MLP入力投影のみ / max | 0.99740 |
| MLP出力投影のみ / max | 0.86000 |
| MLP出力以外の重み付き線形層 / max | 0.99415 |

3aは12 cropで最も精度を保ったattention出力投影のみのINT8 max、3bは画像backboneの4種の重み付き投影をFP8 maxとした。3モデルを同一定義で再計測し、ViTPose-Hの3a・3bは最大座標差が許容外だった。 [3a・3bの定義](../../runs/run-i2297-clipreid-tensorrt/PROTOCOL.md#vit-3モデルのptq設定)・[同一入力の旧条件](000075-run-i2297-clipreid-tensorrt.md)

## 実行精度と追測

TensorRTの8条件すべてでONNXの主要演算、実行タクティクスのオペランドと活性入力をTensorRT Engine Inspectorで照合した。INT8はCLIP-ReID・ViTPose-H・HMR2で12・32・32演算、FP8は48・128・128演算。HMR2のFP16では259演算を照合し、画像入力に依存しない `to_token_embedding/MatMul` が実行層に現れなかったことを記録した。

3モデルの保存済みINT8・FP8 ONNX全6件で、DQの接続先が選定した演算の2つの積入力だけであることを確認した。MatMulと共有する入力から対象外のLayerNormにDQが接続すると、対象行列演算の集合が一致していても入力辺の検査で拒否する。 [照合したONNXと入力辺数](../../runs/run-i2303-vit-tensorrt/validation.json)

監査式が対象演算を認識できない場合、ONNX名とPTQ対象が合わない場合は非ゼロで停止した。INT8のカウンタ収集はHMR2が341件中228件、ViTPose-Hが300件中228件でタイムアウトした。再計測ではplan・入力・出力のdigestを照合し、同じ341件と300件をすべて取得した。途中のCSVは集計から除外した。GVHMRの変更を含むmainにrebaseした後も、保存した8件のinspectorの再集計が変換時と推論時の照合結果に一致した。 [非ゼロ終了と追測](../../runs/run-i2303-vit-tensorrt/validation.json)

## E2Eへの引き継ぎ

単体結果が支持するE2E候補はViTPose-HのTensorRT FP16、HMR2のTensorRT FP16とFP8、CLIP-ReIDの設定変更後のINT8・FP8。CLIP-ReIDの精度判定は同じ2クリップ内に限る。B=1の速度・サンプルされたVRAMは固定入力の単独起動に限る。段間転送、複数モデル常駐、形状やバッチの変化、HMR2特徴差の時系列復元への影響、候補のIDと3D関節の安定性はE2E側で確認する。ViTPose-H量子化の外れ値は、固定cropを指定して再現できる。
