#2305 は `codex/trt-gvhmr` の `1567df1c1` まで進みました。TensorRTの3条件ともビルドは完了しましたが、融合層の精度検査で終了値1となり、計測用planの公開前に停止しました。

融合attentionではBool maskとQ・K・Vを識別し、INT8の残差融合ではQ/DQとタクティクスの両オペランドを照合するよう修正しました。保存済みの9件のInspector出力で、各profileの110行列積を確認しました。FP16は全110演算がFP16、INT8・FP8は各96演算が指定精度で、残る14演算がFP16でした。pytest 75件、Ruff、knowledge検証が通過しました。 [validation.json](https://github.com/Motoki0705/tennis-lab/blob/1567df1c13e9e805b85f979d11e079c8f5dddc4f/knowledge/runs/run-i2305-gvhmr-tensorrt/validation.json)

`i2305-gvhmr-build-and-measure-2-r5`、`i2305-gvhmr-build-and-measure-3a-r5`、`i2305-gvhmr-build-and-measure-3b-r5` を登録しました。再開後は実機の境界検査と145・177フレームの測定を照合し、結果をknowledge runノードとPRに反映します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204837885245029317_2026-11-20T08-08-34-754970682Z_3163088_i2305-gvhmr-build-and-measure-2-r5.job","0-8204837884971504911_2026-11-20T08-08-35-028495088Z_3163165_i2305-gvhmr-build-and-measure-3a-r5.job","0-8204837884702239052_2026-11-20T08-08-35-297760947Z_3163227_i2305-gvhmr-build-and-measure-3b-r5.job"]}}