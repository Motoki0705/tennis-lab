正式レビューのmajor 2件を修正し、[PR #2334](https://github.com/Motoki0705/tennis-lab/pull/2334)の本文を更新しました。`codex/trt-vit-pose-hmr2` に `f59ff9c49494786f281004106215c4e9f5718cd1` をpush済みです。

- **CLIP-ReIDの精度判定**　設定選定に使った12 cropをqueryとgalleryの両方から外し、同じ2クリップの残り224件で再集計しました。cosine P05は3aが **0.999602**、3bが **0.995416**。FP32とのID一致は両方 **224/224** で、レビュー値と一致しました。全236件は旧条件との同一入力比較として保存し、224件の判定をrunノード・summary・PROTOCOLに反映しました。[集計データ](https://github.com/Motoki0705/tennis-lab/blob/f59ff9c49494786f281004106215c4e9f5718cd1/knowledge/runs/run-i2303-vit-tensorrt/metrics.json)
- **PTQのDQ接続検査**　明示レシピでは、TensorRTビルド前にDQの接続先を演算と入力位置で照合し、対象外への接続があれば非ゼロで停止します。MatMulとLayerNormの共有入力を使ったCPU回帰テストを追加し、ONNX Runtimeで再現した混入も拒否することを確認しました。保存済みの量子化ONNXは6件すべてで、計760入力辺が検査を通りました。[検証記録](https://github.com/Motoki0705/tennis-lab/blob/f59ff9c49494786f281004106215c4e9f5718cd1/knowledge/runs/run-i2303-vit-tensorrt/validation.json)

関連pytestは**76件成功（今回の増分4件、5.81秒）**。pre-commitを実行し、hook対象外の変更Python 3ファイルはRuffで直接検証しました。knowledge検証はエラー・警告0件です。再集計とONNX検査はCPUで実施しました。E2E評価と修正HEADのPython CIは未確認です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2334,"head":"f59ff9c49494786f281004106215c4e9f5718cd1","resume_on_success":false}}