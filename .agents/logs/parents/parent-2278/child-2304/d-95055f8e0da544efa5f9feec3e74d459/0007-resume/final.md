GridSample版の正式な条件2は全指標を取得しました。中央値56.98 ms、P95 57.32 ms、17.54枚/秒で、中央値はPyTorch FP32の4.71倍です。nsysとncuは561 kernelで一致しました。IoU P05は0.9325、score絶対差P95は0.0173で、どちらも暫定基準を満たしませんでした。欠落は3件、追加は2件です。 [計測記録](https://github.com/Motoki0705/tennis-lab/blob/37c20deb365fa2fdf34a19f1364ac332aae6e815/knowledge/runs/run-i2304-dino-tensorrt/metrics.json)

プラグイン版はexport時に取得できなかったshapeを、固定入力のtraceから渡すように修正しました。INT8・FP8は、校正対象から外れた18演算が同じ共有重みを直接参照するように修正しました。FP8ではModelOptの対応範囲に合わせ、Swinのshifted-window attentionの24行列積を明示的にFP16に保ちます。CPUテストは89件通過しました。 [試行記録](https://github.com/Motoki0705/tennis-lab/blob/37c20deb365fa2fdf34a19f1364ac332aae6e815/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

`i2304-dino-plugin-2-v5`、`i2304-dino-grid-3a-v5`、`i2304-dino-grid-3b-v5` を登録しました。再開時にプラグイン版との全体比較、PTQの実行精度と全指標を照合し、knowledgeに統合します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204832990778267463_2026-11-20T09-30-09-221732536Z_3677011_i2304-dino-plugin-2-v5.job","0-8204832990272178706_2026-11-20T09-30-09-727821293Z_3677159_i2304-dino-grid-3a-v5.job","0-8204832989781990368_2026-11-20T09-30-10-218009631Z_3677236_i2304-dino-grid-3b-v5.job"]}}