代表 `clip_037` は両条件で完走しました。CLIの処理時間はCUDA計時同期なしで739.79秒、同期ありで899.28秒（+21.56%）でした。Peak VRAMは両条件で1.76 GiB、Peak RSSは最大3.67 GiBでした。38件の成果物参照と318ファイルのSHA-256が一致しました。 [代表計測の記録](https://github.com/Motoki0705/tennis-lab/blob/fef1d626dd0c40ea8ce69705eda777fec4801021/knowledge/runs/run-i2363-trt-player-video/qualification-v2.json)

GVHMRは24 shapeでONNX書換え前後がビット一致し、実GPUの境界6 shapeで有限性とFP16層精度を確認しました。代表クリップではB=4、L=145で動作しました。

全件計測に備え、完了したspanをディスクへ退避し、同じ集計と全spanを最終記録へ出力するようにしました。関連テスト91件とpre-commitが通過しています。ブランチは `codex/parent-2278-issue-2363-primary`、HEADは `fef1d626d` です。

同じ5エンジンで全69クリップの実行を1ジョブとして共有キューへ登録しました。ジョブ完了後に、完走状況、資源ピーク、CPUボトルネック、品質を確認し、可視化とPRにまとめます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204644184801783846_2026-11-22T13-56-55-198216153Z_507741_i2363-trt-all-v1.job"]}}