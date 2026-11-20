#2303 の第3回は、6件中3件が全指標を取得し、1件がGPUカウンタの取得途中で時間切れ、2件が実行精度の照合で停止しました。結果と修正を `codex/trt-vit-pose-hmr2` の `4c585cf199f52226381ae94c8e50edd61bd584d5` に記録しました。

- **CLIP-ReID INT8** はattention出力投影のみの量子化で cosine P05 **0.99960**、FP32とのID一致 **236/236** でした。
- **ViTPose-H FP8** は中央値 **7.73 ms**、座標差P95 **0.284 px** でした。最大差 **29.06 px** は暫定上限5 pxを超えました。
- **HMR2** はINT8・FP8とも単体精度の暫定基準を満たしました。INT8は中央値 **6.57 ms** で、Nsight Computeが341カーネル中228件で時間切れになりました。[条件別の実測](https://github.com/Motoki0705/tennis-lab/blob/4c585cf199f52226381ae94c8e50edd61bd584d5/knowledge/runs/run-i2303-vit-tensorrt/metrics.json)

精度検査を修正し、保存済みinspectorでViTPose-HのINT8投影32演算とHMR2のFP16 259演算を確認しました。pytest **55件**、Ruff、knowledge検証が成功しました。[失敗と再照合の記録](https://github.com/Motoki0705/tennis-lab/blob/4c585cf199f52226381ae94c8e50edd61bd584d5/knowledge/runs/run-i2303-vit-tensorrt/validation.json)

**投入した3件**は `i2303-vitpose-3a-r4`、`i2303-hmr2-2-r4`、`i2303-hmr2-3a-ncu-r4` です。HMR2 INT8の追測には同じエンジンと代表入力を使い、取得上限を30分に延ばしました。再開後は3件の結果を照合し、5条件の比較表とknowledgeのrunノードを完成させてPRを作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204838163145463422_2026-11-20T08-03-56-854536577Z_3129954_i2303-vitpose-3a-r4.job","0-8204838162888051058_2026-11-20T08-03-57-111948941Z_3130036_i2303-hmr2-2-r4.job","0-8204838162611562645_2026-11-20T08-03-57-388437354Z_3130100_i2303-hmr2-3a-ncu-r4.job"]}}