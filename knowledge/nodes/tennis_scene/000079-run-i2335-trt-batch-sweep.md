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
status: done
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
  timing_accuracy_nsys_cases: 37
  historical_timing_accuracy_nsys_cases: 2
  planned_cases: 43
  ncu_with_matching_nsys: 9
  planned_ncu_cases: 11
  gvhmr_trt_fp16_batched_build_failures: 10
  clipreid_b16:
    fp32_images_per_second: 356.37
    pytorch_fp16_images_per_second: 1191.40
    trt_fp16_images_per_second: 2358.36
    trt_int8_images_per_second: 2372.23
    trt_fp8_images_per_second: 2958.10
    trt_fp8_id_matches: 223
  gvhmr_b4:
    l145_fp32_sequences_per_second: 242.92
    l177_fp32_sequences_per_second: 232.84
    l145_fp16_incam_joint_p95_mm: 8.52
    l177_fp16_incam_joint_p95_mm: 8.64
artifacts:
  run_dir: knowledge/runs/run-i2335-trt-batch-sweep
  metrics: knowledge/runs/run-i2335-trt-batch-sweep/metrics.json
  validation: knowledge/runs/run-i2335-trt-batch-sweep/validation.json
  legacy_gvhmr_qdq: knowledge/runs/run-i2335-trt-batch-sweep/legacy-gvhmr-qdq.json
parents:
- run-i2297-clipreid-tensorrt
- run-i2303-vit-tensorrt
- run-i2305-gvhmr-tensorrt
relations: []
papers: []
tags: []
---

## CLIP-ReID

条件0はPyTorch FP32、1はPyTorch FP16、2はTensorRT FP16、3aはTensorRT INT8、3bはTensorRT FP8。精度は各バッチを同じ固定入力のPyTorch FP32 B=1に照合した。3a・3bの表は設定選定用12件をqueryとgalleryから除いた224件。全236件の値も集計に残した。Peak VRAMはPyTorch allocatorのallocated peakと、NVMLでサンプルしたデバイスのidleからの最大増加量を示す。

| 条件 | B | 中央値 / P95 (ms) | crop/s | allocator / NVML増分 (MiB) | nsys kernel | cosine P05 | ID一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 6.10 / 7.11 | 159 | 369.7 / 581.1 | 188 | 1.000000 | 236/236 |
| 0 | 4 | 12.80 / 16.35 | 302 | 393.7 / 599.1 | 201 | 1.000000 | 236/236 |
| 0 | 8 | 22.70 / 23.01 | 353 | 419.0 / 641.1 | 202 | 1.000000 | 236/236 |
| 0 | 16 | 45.13 / 45.35 | 356 | 476.5 / 717.1 | 201 | 1.000000 | 236/236 |
| 0 | 32 | 90.73 / 91.16 | 353 | 590.9 / 885.1 | 201 | 1.000000 | 236/236 |
| 1 | 1 | 3.87 / 7.00 | 233 | 206.0 / 383.1 | 166 | 0.999997 | 236/236 |
| 1 | 4 | 4.74 / 9.70 | 704 | 216.9 / 387.1 | 191 | 0.999997 | 236/236 |
| 1 | 8 | 7.11 / 10.52 | 987 | 231.7 / 407.1 | 191 | 0.999997 | 236/236 |
| 1 | 16 | 13.18 / 13.83 | 1,191 | 260.0 / 461.1 | 189 | 0.999997 | 236/236 |
| 1 | 32 | 28.11 / 28.27 | 1,147 | 317.3 / 537.1 | 189 | 0.999997 | 236/236 |
| 2 | 1 | 1.82 / 2.14 | 536 | 0.2 / 415.1 | 114 | 0.999987 | 236/236 |
| 2 | 4 | 2.49 / 5.95 | 1,279 | 0.8 / 415.1 | 114 | 0.999987 | 236/236 |
| 2 | 8 | 3.91 / 4.38 | 1,970 | 2.3 / 435.1 | 114 | 0.999987 | 236/236 |
| 2 | 16 | 6.64 / 7.50 | 2,358 | 5.3 / 435.1 | 114 | 0.999987 | 236/236 |
| 2 | 32 | 13.79 / 13.89 | 2,341 | 8.3 / 435.1 | 114 | 0.999987 | 236/236 |
| 3a | 1 | 1.90 / 2.26 | 516 | 0.4 / 397.1 | 127 | 0.999594 | 224/224 |
| 3a | 4 | 2.52 / 2.94 | 1,548 | 1.5 / 417.1 | 127 | 0.999594 | 224/224 |
| 3a | 8 | 4.16 / 4.71 | 1,868 | 4.5 / 417.1 | 127 | 0.999594 | 224/224 |
| 3a | 16 | 6.81 / 6.92 | 2,372 | 10.6 / 417.1 | 127 | 0.999594 | 224/224 |
| 3a | 32 | 14.21 / 14.32 | 2,283 | 16.7 / 429.1 | 127 | 0.999594 | 224/224 |
| 3b | 1 | 1.47 / 2.13 | 614 | 0.4 / 277.1 | 115 | 0.995425 | 223/224 |
| 3b | 4 | 1.92 / 2.15 | 2,046 | 1.5 / 297.1 | 115 | 0.995425 | 223/224 |
| 3b | 8 | 3.09 / 3.22 | 2,569 | 4.5 / 297.1 | 115 | 0.995425 | 223/224 |
| 3b | 16 | 5.40 / 5.50 | 2,958 | 10.6 / 297.1 | 115 | 0.995425 | 223/224 |
| 3b | 32 | 11.05 / 11.15 | 2,893 | 16.7 / 309.1 | 115 | 0.995425 | 223/224 |

B=16のFP8は2,958 crop/sで、同じBのTensorRT FP16の2,358 crop/sより25.4%多かった。3条件のTensorRTはB=32でthroughputが下がり、バッチlatencyも増えた。FP8のIDは全バッチで同じ1件が変化した。該当queryは `clip_000/cam3/frame 260/player 1` で、近傍IDは1から2になった。cosine P05は0.995425、ID一致率は99.55%で暫定許容差内。FP8採否は、人物対応への影響と処理単位を下流E2Eで確認して判断する。 [全236件と224件の指標](../../runs/run-i2335-trt-batch-sweep/metrics.json)

3つのTensorRT planはB=1/16/32の単一profileを使った。min・opt・maxでONNXの主要演算74個を照合し、B=0と33を終了値1で拒否した。条件2はFP16が74個、3aはFP16が62個とINT8が12個、3bはFP16が26個とFP8が48個。 [profileと拒否ログ](../../runs/run-i2335-trt-batch-sweep/validation.json)

## GVHMR

フレーム数が同じ4人物系列を束ね、145と177を独立に測った。関節差は同じPyTorch FP32 B=1に対するCOCO-17のP95。計測したPyTorchの全12ケースで長さとmaskが一致した。条件2のB=1は #2305 のB=1固定・L=117/145/177可変のplanで測った既存値。入力lock、人物の並び、復号済みFP32参照のSHA-256を今回と照合した。 [#2305の計測](../../runs/run-i2305-gvhmr-tensorrt/metrics.json)

| L | 条件 | B | 中央値 / P95 (ms) | 系列/s | frames/s | allocator / NVML増分 (MiB) | nsys kernel | カメラ / 重力整列関節差 (mm) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 145 | 0 | 1 | 15.25 / 23.08 | 59.3 | 8,599 | 205.6 / 391.1 | 730 | 0.000 / 0.000 |
| 145 | 0 | 2 | 15.55 / 24.33 | 115.9 | 16,801 | 210.6 / 393.1 | 713 | 0.005 / 1.360 |
| 145 | 0 | 4 | 15.25 / 21.82 | 242.9 | 35,223 | 220.3 / 407.1 | 713 | 0.004 / 1.397 |
| 145 | 1 | 1 | 15.28 / 22.54 | 62.2 | 9,015 | 119.4 / 297.1 | 701 | 8.654 / 24.050 |
| 145 | 1 | 2 | 14.88 / 40.45 | 103.6 | 15,016 | 121.9 / 301.1 | 712 | 8.503 / 21.422 |
| 145 | 1 | 4 | 15.80 / 23.29 | 221.7 | 32,151 | 126.7 / 303.1 | 712 | 8.523 / 23.961 |
| 145 | 2 | 1 | 2.51 / 3.17 | 393.6 | 57,068 | 0.3 / 263.1 | 135 | 8.614 / 23.499 |
| 145 | 2 | 2 | ビルド不可 | — | — | — | — | — |
| 145 | 2 | 4 | ビルド不可 | — | — | — | — | — |
| 177 | 0 | 1 | 16.70 / 38.93 | 44.8 | 7,936 | 206.5 / 393.1 | 867 | 0.000 / 0.000 |
| 177 | 0 | 2 | 15.28 / 23.01 | 123.9 | 21,926 | 212.5 / 397.1 | 745 | 0.006 / 0.534 |
| 177 | 0 | 4 | 16.37 / 23.58 | 232.8 | 41,213 | 224.4 / 407.1 | 745 | 0.016 / 1.594 |
| 177 | 1 | 1 | 15.10 / 19.30 | 63.4 | 11,227 | 119.9 / 299.1 | 732 | 8.550 / 20.127 |
| 177 | 1 | 2 | 15.06 / 22.17 | 126.6 | 22,401 | 123.0 / 303.1 | 744 | 8.451 / 16.474 |
| 177 | 1 | 4 | 15.85 / 25.54 | 216.1 | 38,256 | 128.9 / 307.1 | 745 | 8.637 / 17.756 |
| 177 | 2 | 1 | 2.50 / 2.98 | 394.2 | 69,780 | 0.4 / 263.1 | 135 | 8.721 / 18.325 |
| 177 | 2 | 2 | ビルド不可 | — | — | — | — | — |
| 177 | 2 | 4 | ビルド不可 | — | — | — | — | — |

PyTorch FP32のバッチ間のカメラ座標関節差はP95最大0.017 mmだった。PyTorch FP16の関節差はカメラ座標で8.45–8.65 mm、重力整列座標で16.47–24.05 mmとなり、座標系によって暫定10 mm基準をまたいだ。 [精度の内訳](../../runs/run-i2335-trt-batch-sweep/metrics.json)

今回のRTX 5060 TiとTensorRT 10.14.1.48.post1では、GVHMRのFP16 B>1はビルド不可だった。合計10件で変換の子プロセスは−11（SIGSEGV）、queueジョブは1で終了した。B>1のplanが生成されず、速度・VRAM・nsys・精度、B=4のncu、層精度と範囲外shape拒否の実機確認は未取得。 [試行ごとの終了状態・ログ・SHA-256](../../runs/run-i2335-trt-batch-sweep/validation.json)

| 試行 | B / L | 件数 | 結果 | verboseログの最後のTiming Runner |
|---|---|---:|---|---|
| 可変B・可変L | B=1/4/4、L=145/145/177。初回と動的mask修正後 | 2 | −11 / queue 1 | — |
| 可変B・固定L、既定の最適化 | B=1/4/4、L=145と177を別々に作成 | 2 | −11 / queue 1 | — |
| 可変B・固定L、最適化レベル0 | B=1/4/4、L=145と177を別々に作成 | 2 | −11 / queue 1 | `ForeignNode[/network/Unsqueeze_7.../network/ScatterND]` |
| 固定B・固定L、最適化レベル3 | B=2・4 × L=145・177を別々に作成 | 4 | −11 / queue 1 | `ForeignNode[/network/Unsqueeze_3.../network/ScatterND]` |

固定shapeのONNXでは `/network/Unsqueeze_3` の入力は `length`、`/network/ScatterND` の出力は `pred_cam`。B=2、L=145のパーサーログではScatterNDが登録され、FP16データ `(2,145,3)`、INT64インデックス `(2,145,1,3)`、FP16更新値 `(2,145,1)` を受け取っていた。全4件でMyelinのTiming Runnerとcompiler backendのメッセージに続いて `build_serialized_network` 内で停止した。SIGSEGVの原因となった内部演算の特定には至っていない。 [固定shapeログの抜粋と入出力](../../runs/run-i2335-trt-batch-sweep/validation.json)

パイプラインで人物を束ねずに処理する場合、#2305のTensorRT FP16 B=1固定planを人物ごとに使う構成を評価できる。検証済みの系列長profileはmin/opt/max=117/145/177。実測は145・177で393.6・394.2系列/sだった。標準パイプラインでの速度と下流精度はE2Eで確認する。 [B=1の範囲と検査](../../runs/run-i2305-gvhmr-tensorrt/validation.json)

### #2305で使ったPTQグラフのQ/DQ範囲

条件3a（INT8）と3b（FP8）のONNXを当時の変換証跡とSHA-256で照合し、両評価長が同じエンジンを使ったことを確認した。CPUで `onnx.checker` とmainの `verify_qdq_scope` を適用した。両条件とも指定した96積すべての2オペランドにDQがあり、指定外への直接接続はShapeの24ノード・24入力だった。Shapeは入力の次元を参照する。Add、LayerNormalization、対象外のMatMul・Gemm、graph outputへの直接接続は各0件。現行のscope検査は両グラフを拒否した。 [演算種別ごとの件数と元グラフのSHA-256](../../runs/run-i2335-trt-batch-sweep/legacy-gvhmr-qdq.json)

## GPU内部指標

DRAM R/W、Tensor Core active、stallは先行のB=1 → 今回の代表Bの順で示す。stallはlong scoreboard、short scoreboard、LG throttle、MIO throttleをGPU kernel時間で加重した割合。B=1は #2297・#2303・#2305 の計測値。TensorRTのB=1専用planと掃引用planの比較にはタクティクスの差も含む。内部指標を採用した代表Bの全9件でncuとnsysのカーネル数が一致した。

| モデル | 条件 | 代表B | 代表Bのkernel | DRAM read (MB) | DRAM write (MB) | Tensor Core active (%) | stall (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIP-ReID | 0 | 16 | 201 | 483.8 → 3197.4 | 0.0 → 420.9 | 1.4 → 2.6 | 4.6 / 7.9 / 2.4 / 31.2 → 23.6 / 5.6 / 0.6 / 11.2 |
| CLIP-ReID | 1 | 16 | 189 | 191.2 → 1552.5 | 0.0 → 14.9 | 19.2 → 28.8 | 11.8 / 3.9 / 1.4 / 1.6 → 27.2 / 2.2 / 0.5 / 1.4 |
| CLIP-ReID | 2 | 16 | 114 | 161.1 → 976.1 | 0.0 → 4.2 | 29.9 → 51.0 | 9.5 / 3.7 / 0.4 / 1.6 → 22.5 / 2.3 / 0.9 / 1.6 |
| CLIP-ReID | 3a | 16 | 127 | 155.4 → 973.6 | 0.0 → 11.9 | 28.9 → 48.7 | 9.1 / 4.1 / 0.4 / 1.6 → 24.4 / 2.7 / 0.9 / 1.6 |
| CLIP-ReID | 3b | 16 | 115 | 76.0 → 725.1 | 0.0 → 9.1 | 17.1 → 31.1 | 16.6 / 7.2 / 0.9 / 1.9 → 13.4 / 3.5 / 0.8 / 1.3 |
| GVHMR L=145 | 0 | 4 | 713 | 275.5 → 802.3 | 0.0 → 0.0 | 0.0 → 0.0 | 8.8 / 8.5 / 2.4 / 21.5 → 6.5 / 9.2 / 2.2 / 7.2 |
| GVHMR L=145 | 1 | 4 | 712 | 106.7 → 388.0 | 0.0 → 0.0 | 7.3 → 12.6 | 18.6 / 8.6 / 0.6 / 3.4 → 13.6 / 4.3 / 1.9 / 2.2 |
| GVHMR L=145 | 2 | 4 | — | 71.3 → — | 0.0 → — | 20.2 → — | 14.2 / 5.1 / 0.6 / 1.8 → — |
| GVHMR L=177 | 0 | 4 | 745 | 345.2 → 955.4 | 0.0 → 0.1 | 0.0 → 0.0 | 7.1 / 7.3 / 1.0 / 22.5 → 7.3 / 6.2 / 3.9 / 7.6 |
| GVHMR L=177 | 1 | 4 | 745 | 150.7 → 493.2 | 0.0 → 0.0 | 6.4 → 13.2 | 20.2 / 8.0 / 0.7 / 1.6 → 13.9 / 4.2 / 1.9 / 2.3 |
| GVHMR L=177 | 2 | 4 | — | 71.6 → — | 0.0 → — | 19.3 → — | 14.2 / 5.5 / 0.6 / 1.6 → — |

CLIP-ReIDのB=16でDRAM readはFP16の61.0 MB/cropからFP8で45.3 MB/crop、long scoreboardは22.5%から13.4%になった。Tensor Core activeは同じ比較で51.0%から31.1%になった。観測値はkernel時間で加重した比率で、DRAM帯域飽和の有無はこの計測からは判定できない。PyTorch FP16とTensorRT FP16のB=16は `retry-2` のncuが時間切れで、nsysの189件と114件のうち119件と66件で止まった。途中CSVだけの値は採用せず、追測した70件と48件を足し、順番・grid・blockの全件照合後に集計した。GVHMRのncu未取得2件はTensorRT FP16のB=4、L=145と177。変換が止まったためmeasurement.jsonがなく両方ともncu起動前に終了し、カーネル数の比較に至らなかった。 [B=1の値と計測区間](../../runs/run-i2335-trt-batch-sweep/metrics.json)・[終了ログの記録](../../runs/run-i2335-trt-batch-sweep/validation.json)

標準パイプラインでのCLIP-ReIDとGVHMRの採用判断は、下流の人物対応と時系列の精度、実行速度のE2E確認に委ねる。 [標準パイプラインの判断記録](../../summary.md#標準pipelinetennis_scene)
