[PR #2327](https://github.com/Motoki0705/tennis-lab/pull/2327) のドラフトを解除しました。`codex/trt-gvhmr` のHEAD `4fe8626cf8ce0ecf81703941e582d1d4898f57d9` で [Python CIの成功](https://github.com/Motoki0705/tennis-lab/actions/runs/44727932191/job/136271965550) を照合し、解除後も同じHEADであることを確認しました。

TensorRT FP16のdenoiser中央値は145フレームで **2.509 ms**、177フレームで **2.498 ms** でした。PyTorch FP32比でそれぞれ **7.55倍、6.62倍** です。カメラ座標のCOCO-17関節差P95は **8.61／8.72 mm** で、暫定許容差の10 mm以内でした。

### 受入条件

| 条件 | 達成と証拠 |
|---|---|
| 連続系列と全入力の固定 | 達成。評価は `clip_037` の145フレームと `clip_000` の177フレームを各4系列、PTQは `clip_027` の117フレームを4系列。再採取でdigestが一致し、評価とPTQの収録元フレーム区間に重複がないことを確認しました。[系列一覧・採取手順](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/README.md) · [tensor lock](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/tensor-lock.json) |
| 動的profileと範囲外拒否 | 達成。B=1、min/opt/maxは117/145/177。3エンジンともこの3長で実行でき、116・178はTensorRTが拒否して終了値1を返しました。[各テンソルのprofile](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/shape-profile.json) · [境界ログ](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json) |
| 5条件の計測とknowledge記録 | 達成。評価2区分の全10組を実測し、全組でnsysとncuのkernel数が一致しました。各条件の1,288フレームでlengthとmaskが一致しました。[metrics.json](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/metrics.json) · [knowledge runノード](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/nodes/tennis_scene/000076-run-i2305-gvhmr-tensorrt.md) |
| 失敗時の停止と指定精度の照合 | 達成。変換・精度検査で失敗した試行は非ゼロ終了を記録しました。成功したエンジンは3長すべてでInspectorを照合し、FP16は主要110演算、INT8・FP8はそれぞれ指定精度96演算と設定どおりのFP16 14演算を確認しました。[変換・精度検査ログ](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json) |

### 実測

| 系列長 | 条件 | 中央値 / P95 (ms) | 系列[local path] / フレーム[local path] | Peak VRAM (MiB) | DRAM R/W (B) | Kernel数 | 最大stall | Tensor Core active |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 145 | PyTorch FP32 | 18.942 / 25.603 | 50.9 / 7,382 | 652.1 | 275,522,304 / 0 | 730 | `mio_throttle` 21.5% | 0.0% |
| 145 | PyTorch FP16 | 14.650 / 16.872 | 67.5 / 9,780 | 558.1 | 106,659,072 / 0 | 701 | `long_scoreboard` 18.6% | 7.3% |
| 145 | TensorRT FP16 | 2.509 / 3.170 | 393.6 / 57,068 | 524.1 | 71,316,992 / 0 | 135 | `long_scoreboard` 14.2% | 20.2% |
| 145 | TensorRT INT8 | 2.332 / 3.171 | 411.6 / 59,685 | 494.1 | 39,428,352 / 0 | 150 | `long_scoreboard` 13.8% | 11.3% |
| 145 | TensorRT FP8 | 2.328 / 3.099 | 411.8 / 59,706 | 492.1 | 40,694,016 / 0 | 151 | `long_scoreboard` 12.0% | 10.1% |
| 177 | PyTorch FP32 | 16.541 / 18.870 | 59.5 / 10,528 | 654.1 | 345,229,056 / 0 | 867 | `mio_throttle` 22.5% | 0.0% |
| 177 | PyTorch FP16 | 14.115 / 16.458 | 69.2 / 12,248 | 560.1 | 150,749,696 / 512 | 732 | `long_scoreboard` 20.2% | 6.4% |
| 177 | TensorRT FP16 | 2.498 / 2.979 | 394.2 / 69,780 | 524.1 | 71,583,232 / 0 | 135 | `long_scoreboard` 14.2% | 19.3% |
| 177 | TensorRT INT8 | 2.378 / 3.014 | 409.7 / 72,508 | 494.1 | 40,538,624 / 0 | 150 | `long_scoreboard` 12.6% | 11.0% |
| 177 | TensorRT FP8 | 2.512 / 3.063 | 391.7 / 69,330 | 492.1 | 42,071,040 / 0 | 151 | `long_scoreboard` 11.6% | 9.9% |

単体精度は同じ入力のPyTorch FP32に対するカメラ座標での差のP95です。各セルは系列長145／177の順です。

| 条件 | body_pose (°) | global_orient (°) | betas絶対差 | transl L2 (mm) | COCO-17 L2 (mm) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| PyTorch FP16 | 0.062 / 0.063 | 0.041 / 0.047 | 0.00108 / 0.00138 | 8.51 / 8.56 | 8.65 / 8.55 |
| TensorRT FP16 | 0.087 / 0.085 | 0.065 / 0.069 | 0.00161 / 0.00184 | 8.49 / 8.68 | 8.61 / 8.72 |
| TensorRT INT8 | 1.848 / 2.116 | 0.848 / 1.098 | 0.05850 / 0.04046 | 92.09 / 80.19 | 94.54 / 81.96 |
| TensorRT FP8 | 1.606 / 1.815 | 0.612 / 0.717 | 0.03472 / 0.05099 | 35.56 / 67.25 | 36.95 / 68.93 |

ローカルではpytest **75件成功（追加17件、34.12秒）**。変更したPythonのRuff check・formatとknowledge検証も通過しました。

### E2Eへの引き継ぎ

速度表はdenoiser区間の計測です。E2Eでは前処理・入力転送・身体配置・I/Oを含む総時間と、三角測量で配置した後の関節・位置・向き・有効maskの照合が残っています。TensorRT FP16でも重力整列座標の関節差P95は23.50／18.32 mmでした。

PTQは計測時に #2303 がmainへ未マージだったため、#2297の `max`・ORT CPUを採用しました。117フレームの校正系列では評価系列が使う局所attentionを通りません。この設定のINT8・FP8は両区分で回転と関節差の暫定許容値を超えました。[PTQ設定](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/ptq.json)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}