プラグイン版FP16とFP8の計測をknowledgeに反映しました。 [計測記録](https://github.com/Motoki0705/tennis-lab/blob/dccf525b98c0072c4ffe64f4d7c47f084d850adb/knowledge/runs/run-i2304-dino-tensorrt/metrics.json)

| 条件 | 中央値 / P95 | IoU P05 | 欠落 / 追加 |
|---|---:|---:|---:|
| FP16 GridSample | 56.98 / 57.32 ms | 0.9325 | 3 / 2 |
| FP16 プラグイン | 41.47 / 41.82 ms | 0.9310 | 4 / 4 |
| FP8 GridSample | 52.55 / 52.84 ms | 0.9132 | 5 / 11 |

プラグイン版は全体モデルでGridSample版より27%短いレイテンシでした。FP8はnsysとncuで608 kernelが一致しました。

INT8は、融合層のFP16残差をGEMMの入力として扱った検査で停止していました。修正後、保存済みinspectorで292演算のINT8と24演算のFP16を照合できました。共有harnessのmain更新も取り込み、CPUテスト96件が通過しました。 [試行記録](https://github.com/Motoki0705/tennis-lab/blob/dccf525b98c0072c4ffe64f4d7c47f084d850adb/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

校正済みONNXを使う `i2304-dino-grid-3a-v6` を登録しました。再開時にINT8の実行精度と計測値を照合し、knowledgeノードとPRを仕上げます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204822146448313429_2026-11-20T12-30-53-551686570Z_638131_i2304-dino-grid-3a-v6.job"]}}