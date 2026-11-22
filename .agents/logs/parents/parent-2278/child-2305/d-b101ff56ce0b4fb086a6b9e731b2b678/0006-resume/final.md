5条件×2系列長の実測を完了し、[ドラフトPR #2327](https://github.com/Motoki0705/tennis-lab/pull/2327) を作成しました。ブランチは `codex/trt-gvhmr`、HEADは `4fe8626cf` です。pytest 75件、Ruff、knowledge検証が通過しました。

- 入力は評価8系列とPTQ 4系列を固定しました。再採取でdigestが一致し、評価とPTQの収録元区間に重複がないことを確認しました。 [tensor lock](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/tensor-lock.json)
- TensorRTの3条件すべてで117・145・177フレームを実行し、116・178フレームを終了値1で拒否しました。各shapeの層精度と、失敗した試行の非ゼロ終了を記録しました。 [validation.json](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json)
- TensorRT FP16の中央値は145／177フレームで2.51／2.50 ms、PyTorch FP32は18.94／16.54 msでした。FP16のカメラ座標の関節差P95は8.61／8.72 mm、重力整列座標では23.50／18.32 mmです。INT8・FP8のカメラ座標差は暫定許容値を超えました。 [全条件の比較](https://github.com/Motoki0705/tennis-lab/blob/4fe8626cf8ce0ecf81703941e582d1d4898f57d9/knowledge/nodes/tennis_scene/000076-run-i2305-gvhmr-tensorrt.md)

初回キュー登録から最終計測まで8時間29分、ジョブ実行時間の合計は92.6分でした。対象HEADの `Python` CI成功を確認したターンで、ドラフト解除と最終報告を行います。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2327,"head":"4fe8626cf8ce0ecf81703941e582d1d4898f57d9","resume_on_success":true,"next_step":"対象HEADのPython CI成功を照合し、ドラフトを解除して最終報告する"}}