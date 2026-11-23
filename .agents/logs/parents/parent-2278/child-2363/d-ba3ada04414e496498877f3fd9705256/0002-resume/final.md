初回ジョブは5モデルのエンジン準備を完了しました。GVHMRは24 shapeでONNX書換え前後がビット一致し、B=1/4 × L=2/177/1055の6 shapeでTensorRT出力の有限性と層精度を確認しました。 [初回の証拠とエンジン識別](https://github.com/Motoki0705/tennis-lab/blob/80db8e57a0bfe3642aa636e12f3da87a1fdd0f5f/knowledge/runs/run-i2363-trt-player-video/qualification-v1.json)

代表クリップはHydraの出力先指定で推論前に停止しました。出力パスと検証JSONの上書きを修正し、`codex/parent-2278-issue-2363-primary` の `80db8e57a` にコミットしました。回帰テスト16件、ruff、mypyが通過しました。

同じ5エンジンのSHA-256を照合し、GVHMRの検証記録の再取得と `clip_037` のCUDA計時同期あり・なしの計測を再投入しました。全69クリップの実行は、代表クリップの結果と資源使用量を確認した後に進めます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204646797739929593_2026-11-22T13-13-22-260070406Z_4106821_i2363-trt-qualification-v2.job"]}}