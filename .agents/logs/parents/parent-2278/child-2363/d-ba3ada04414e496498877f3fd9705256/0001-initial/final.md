#2363 のCPU計測、GVHMRの複数人物バッチ対応、全件評価の実行手順を実装しました。ブランチは `codex/parent-2278-issue-2363-primary`、HEADは `d640589ef` です。

CPU側の158ケースが成功しました。成果物のバイト列・artifact ID・execution keyの一致と、小型GVHMRの3 shapeでONNX書換え前後のビット一致を確認しました。CLIの2ケースは並列実行で既存の30秒制限を超え、単独実行では成功しました。実装コミットのruff・mypyは通過しています。

ViTPose-H 8 crop、HMR2 16、DINO GridSample 1、CLIP-ReID 16、GVHMR 4を指定し、承認済みの共有コートで実GPU検証を共有キューへ登録しました。このジョブでGVHMRの層精度・境界の有限性と、代表 `clip_037` のCUDA計時同期あり・なしの処理時間、Peak VRAM・RSSを確認します。 [評価条件と再現手順](https://github.com/Motoki0705/tennis-lab/blob/d640589ef2913455aa0c294531a6cb8f129d7bcd/knowledge/runs/run-i2363-trt-player-video/README.md)

代表クリップの結果を照合した後、全69クリップの一括実行、品質分析、可視化、PR作成に進みます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204647824645612917_2026-11-22T12-56-15-354387082Z_3961356_i2363-trt-qualification-v1.job"]}}