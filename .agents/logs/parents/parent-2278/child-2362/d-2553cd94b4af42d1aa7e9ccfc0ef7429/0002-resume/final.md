#2362の初回検証では、5モデルすべてのONNX exportが成功しました。エンジン作成は、TensorRTのprofile設定APIが返す `None` を失敗と判定して停止しました。 [初回の終了状態とONNXのdigest](https://github.com/Motoki0705/tennis-lab/blob/f5bbcb0c2f2953eb28e8f700b62f80081655aad8/knowledge/runs/run-i2362-tensorrt-pipeline/attempts.json)

profileのmin・opt・maxを読み戻して照合するよう修正しました。ONNXで未確定の出力軸も、推論時にモデルが要求するshapeと照合します。CPUテスト57件、pre-commit、knowledge検証が通過し、既定FP32のpublication identityと38 componentの設定は変更前と一致しました。

ブランチ `codex/parent-2278-issue-2362-primary` のHEADは `f5bbcb0c2f2953eb28e8f700b62f80081655aad8` です。既存ONNXからの5モデルのbuild、出力照合、短いクリップのsmokeを再登録しました。実GPUの結果とPR作成は後続ターンに残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204663892298719542_2026-11-22T08-28-27-701280457Z_2349504_i2362-phase2-rebuild.job"]}}