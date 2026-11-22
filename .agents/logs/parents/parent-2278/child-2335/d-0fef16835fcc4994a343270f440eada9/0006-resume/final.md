`Closes #2335` を含む日本語ドラフトPRを作成しました。ブランチは `codex/parent-2278-issue-2335-primary`、HEADは `80d035e807169fe2af99866b25b1c197dde8be6a` です。Python CI成功後にドラフトを解除します。 [PR #2349](https://github.com/Motoki0705/tennis-lab/pull/2349)

新規実測37件と、入力・FP32参照が一致した #2305 の2件で、43ケース中39ケースを記録しました。CLIP-ReIDはcrop/s、GVHMRは系列[local path]。各ケースの中央値・P95・Peak VRAM・nsys・単体精度も記録しています。 [全ケースの実測表](https://github.com/Motoki0705/tennis-lab/blob/80d035e807169fe2af99866b25b1c197dde8be6a/knowledge/nodes/tennis_scene/000079-run-i2335-trt-batch-sweep.md)

| モデル | 条件 | Bの順 | Throughput |
|---|---|---|---|
| CLIP-ReID | PyTorch FP32 | 1 / 4 / 8 / 16 / 32 | 159 / 302 / 353 / 356 / 353 |
| CLIP-ReID | PyTorch FP16 | 同上 | 233 / 704 / 987 / 1,191 / 1,147 |
| CLIP-ReID | TRT FP16 | 同上 | 536 / 1,279 / 1,970 / 2,358 / 2,341 |
| CLIP-ReID | TRT INT8 | 同上 | 516 / 1,548 / 1,868 / 2,372 / 2,283 |
| CLIP-ReID | TRT FP8 | 同上 | 614 / 2,046 / 2,569 / 2,958 / 2,893 |
| GVHMR L=145 | PyTorch FP32 / FP16 | 1 / 2 / 4 | 59.3 / 115.9 / 242.9、62.2 / 103.6 / 221.7 |
| GVHMR L=177 | PyTorch FP32 / FP16 | 1 / 2 / 4 | 44.8 / 123.9 / 232.8、63.4 / 126.6 / 216.1 |
| GVHMR L=145 / 177 | TRT FP16 | 1 / 2 / 4 | B=1は393.6 / 394.2。B=2・4はビルド不可 |

代表バッチのncuは11件中9件でnsysと一致しました。CLIP-ReIDのFP8はB=1→16で、DRAM読込76.0→725.1 MB、書込0.0→9.1 MB、Tensor Core active 17.1→31.1%、long scoreboard 16.6→13.4%でした。B=16のID一致は223/224件です。GVHMR TRT FP16のB=4は両系列長ともビルド段階で止まり、ncuと範囲外拒否・層精度の実機確認は未取得です。 [ncuの比較表と検証記録](https://github.com/Motoki0705/tennis-lab/blob/80d035e807169fe2af99866b25b1c197dde8be6a/knowledge/nodes/tennis_scene/000079-run-i2335-trt-batch-sweep.md#gpu内部指標)

GVHMRは可変profile、最適化レベル0、B=2・4とL=145・177の固定shapeを含む全10試行で、ビルド子が−11、queueが1で終了しました。固定shapeの最後のTiming Runnerは全4件とも `ForeignNode[[local path]]` でした。`pred_cam` を出力するScatterNDの登録は確認できましたが、SIGSEGVの原因となった内部演算は未特定です。人物ごとに処理する場合は、#2305のB=1固定plan（L=117/145/177）がE2E評価の候補です。 [各試行のログ証拠](https://github.com/Motoki0705/tennis-lab/blob/80d035e807169fe2af99866b25b1c197dde8be6a/knowledge/runs/run-i2335-trt-batch-sweep/validation.json)

#2305の旧INT8・FP8 ONNXは、両条件とも指定外のDQ接続がShapeの24件でした。Add・LayerNormalization・対象外の積・graph outputへの直接接続は各0件で、現行の `verify_qdq_scope` は両グラフを拒否しました。 [CPU検査結果](https://github.com/Motoki0705/tennis-lab/blob/80d035e807169fe2af99866b25b1c197dde8be6a/knowledge/runs/run-i2335-trt-batch-sweep/legacy-gvhmr-qdq.json)

#2336には `batch_slices`、`measure_latency`、`build_engine(profile_shapes=...)`、`require_engine_profile`、`profiles.capture` の使い方を残しました。ViTPose-Hはflip込みのエンジン入力2Bと処理量のB cropを区別し、DINOは同一H/Wで束ねます。CPUテスト90件（増分14件）が8.55秒で通り、Ruff、knowledge検証、実測からのJSON再生成も確認しました。標準パイプラインの速度・人物対応・時系列精度はE2Eでの確認が残ります。 [#2336への引き継ぎ](https://github.com/Motoki0705/tennis-lab/blob/80d035e807169fe2af99866b25b1c197dde8be6a/knowledge/runs/run-i2335-trt-batch-sweep/README.md#2336での再利用)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2349,"head":"80d035e807169fe2af99866b25b1c197dde8be6a","resume_on_success":true,"next_step":"ドラフト解除と最終報告"}}