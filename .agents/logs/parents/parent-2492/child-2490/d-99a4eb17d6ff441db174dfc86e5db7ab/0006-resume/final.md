保存済み18条件を各96系列に並べ、元フレーム順と `Pose2DResult` の `(F,17,3)` 形式を確認しました。RTMPose-M TRT16だけで右手首の座標にNaNが1件ありました。現行パイプラインは非有限値を拒否するため、接続前に欠測の扱いを決める必要があります。既存テスト7件は成功し、CI成功済みのHEAD `1dda95b51398b15e9c69d0b4bf2a92ebe7472eeb` のPR本文に証拠を追記しました。 [ドラフトPR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500)

Sapiens2-1B PT32の既存ジョブは実行中で、統合できる新しい予測・変換結果はまだありません。このジョブの終端を待ちます。速度再計測とPareto比較は、#2497の修正版の公開と親の再投入指示待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204471621916598843_2026-11-24T13-52-58-083401156Z_2244865_i2490-sapiens2-1b-pt32-accuracy.job"]}}