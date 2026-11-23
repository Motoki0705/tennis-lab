#2362の5モデルにTensorRT FP16を接続しました。`backend`、`precision`、`engines`、`batch_size`でモデルごとに選択できます。エンジンは `outputs/tensorrt/<model>[local path]<shape>[local path]<build-tag>[local path]` に置き、ONNXとplanのSHA-256、TensorRT版、GPU、入力shape、profileを実行前に照合します。GVHMRのTensorRTはバッチ1に制限しています。 [設定と作成手順](https://github.com/Motoki0705/tennis-lab/blob/3b1e5d60f76059e84ffb30a77d0f2fd582bb603d/src/tennis_scene/pipeline/README.md%23tensorrt-fp16)

- 共有 `.venv` にTensorRT 10.14.1.48.post1、ONNX 1.21.0、ONNX Runtime GPU 1.30.0と推移依存を導入しました。既存185パッケージの版は導入前後で一致し、import、FP16 ONNXの試験export、`uv pip check`が通りました。
- CPU回帰テストは **1,836件成功**（157.72秒）。同じ資産を使い、既定FP32のpublication identityと38 componentの設定が変更前と一致することを確認しました。pre-commitも通過しました。
- ブランチは `codex/parent-2278-issue-2362-primary`、HEADは `3b1e5d60f76059e84ffb30a77d0f2fd582bb603d` です。

5モデルのエンジン作成、Phase 1と同じ指標での出力照合、短いクリップのsmokeを共有GPUキューに登録しました。実GPUの結果は未確認で、PRは結果確認後に作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204676305274100327_2026-11-22T05-01-34-725899672Z_632389_i2362-phase2-qualification.job"]}}