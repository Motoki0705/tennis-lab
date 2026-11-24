PMPose-L/HのBBox指定adapterと、重み取得・ONNX変換・TensorRTビルドの再現手順をコミットしました。著者デモと同じくBBoxから矩形マスクを作り、公式のflip testを2 forwardで実行します。COCO17のconfidenceには期待OKSを使い、presence・visibility・期待OKSなどは23関節分を保存します。 [実装・再現手順](https://github.com/Motoki0705/tennis-lab/blob/5e51efffbde47cda83f2052141e67906eff811ec/knowledge/runs/run-meiji-topdown-pose-comparison/PMPose.md)

- ブランチ `codex/parent-2492-issue-2490-pmpose`、コミット `5e51efffbde47cda83f2052141e67906eff811ec`
- CPUテスト28件成功（新規8件、既存20件、6.67秒）。Ruff・mypy・knowledge検査も通過しました。#2495の固定manifestを照合し、6,844画像・26,276 BBoxを確認しました。
- 公式重みの取得と依存ライブラリのビルドを起動しました。完了後に公式APIとの実重みparityを確認します。PT32・PT16・TRT16の精度、速度、VRAMは未計測です。
- 速度計測の接続点として、#2496の更新で `BatchStages.forward()` が `measure_forward()` に変わっています。primaryの `PoseStages.forward()` から接続する際は、この差分の吸収が必要です。 [#2496の更新](https://github.com/Motoki0705/tennis-lab/pull/2497/commits/d64962a28d81dd7646c7a19b33f88143b67a90f4)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T113813Z_16d5e8631dce_i2490-pmpose-weights","20261124T113813Z_29ea3f805bfa_i2490-pmpose-deps"]}}