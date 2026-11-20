#2305 は `codex/trt-gvhmr` の `a845ef991` で再計測を登録しました。初回7件はすべて、CUDA処理に入る前にHydraのimportエラーで終了値1でした。固定済み入力の検証を採取処理から独立させ、初回ログの識別子と終了値を記録しました。 [validation.json](https://github.com/Motoki0705/tennis-lab/blob/a845ef9912684fdb527eff8b015560cef4602442/knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json)

専用環境で4つのCLIの起動と、実入力145フレームのFP32推論・SMPL-X復号・両座標系のCOCO-17再構成をCPUで確認しました。関連pytestは66件成功し、Ruffとknowledge検証も通過しました。12系列のtensor lockは再採取で一致しています。 [tensor-lock.json](https://github.com/Motoki0705/tennis-lab/blob/a845ef9912684fdb527eff8b015560cef4602442/knowledge/runs/run-i2305-gvhmr-tensorrt/tensor-lock.json)

登録したジョブは次の7件です。

- `i2305-gvhmr-build-2-r2`、`i2305-gvhmr-build-3a-r2`、`i2305-gvhmr-build-3b-r2` は変換・実行精度の検査を行い、成功時に117・145・177の実行と116・178の拒否を検証します。
- `i2305-gvhmr-0-l145-r2`、`i2305-gvhmr-0-l177-r2`、`i2305-gvhmr-1-l145-r2`、`i2305-gvhmr-1-l177-r2` はPyTorchの速度・VRAM・精度・GPU内部指標を収集します。

再開後に終了ログと計測結果を照合し、変換できたTensorRT条件の評価計測へ進みます。GPU実測の確定、knowledge runノード、PRと対象HEADのCIは残っています。