#2305 は `codex/trt-gvhmr` の `2f4114a2b` まで進みました。PyTorchのFP32・FP16は、145・177フレームの4組で速度・Peak VRAM・単体精度を取得し、全組でlengthとmaskが一致しました。FP16のカメラ座標COCO-17誤差はFP32比でP95が8.65 mm／8.55 mmでした。 [metrics.json](https://github.com/Motoki0705/tennis-lab/blob/2f4114a2bbe34e5647a9b1edfed9ec53f74d5dcc/knowledge/runs/run-i2305-gvhmr-tensorrt/metrics.json)

TensorRTの3条件は、ONNXの `Where` で可視maskの座標軸が欠けていたため終了値1でした。変換処理を修正し、回帰テストを含む69件とknowledge検証が通過しました。FP32・177フレームのncuは900秒で打ち切られ、完了したnsysの867 kernelに対して455件まで取得していました。 [validation.json](https://github.com/Motoki0705/tennis-lab/blob/2f4114a2bbe34e5647a9b1edfed9ec53f74d5dcc/knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json)

共有キューに4件を登録しました。

- `i2305-gvhmr-build-2-r3`、`i2305-gvhmr-build-3a-r3`、`i2305-gvhmr-build-3b-r3` は再変換と層精度検査を行い、成功時に117・145・177の実行と116・178の拒否を検証します。
- `i2305-gvhmr-ncu-0-l177-r3` は既存のnsysを参照し、制限2400秒でncuだけを追測します。

再開後に変換・境界検査とncuの結果を照合し、検証を通過したTensorRT条件を評価します。knowledge runノード、PR、対象HEADのCIは残っています。