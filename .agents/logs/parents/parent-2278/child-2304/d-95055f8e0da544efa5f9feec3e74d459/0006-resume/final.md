#2304 の5件を `codex/trt-dino` から共有キューに登録しました。

- `i2304-dino-grid-2-v4` — GridSample版FP16の全指標
- `i2304-dino-plugin-2-v4` — プラグイン版FP16の速度・精度とnsys
- `i2304-dino-export-grid-v4` — PTQ用の元ONNX再生成
- `i2304-dino-grid-3a-v4`、`i2304-dino-grid-3b-v4` — INT8・FP8の変換と全指標

再開時に各ジョブの終了段階、実行層の精度、取得した指標を照合します。GridSample版を正式な条件2とし、プラグイン版との全体比較をknowledgeに記録します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204841827830284670_2026-11-20T07-02-52-169715329Z_2757180_i2304-dino-grid-2-v4.job","0-8204841827569138645_2026-11-20T07-02-52-430861354Z_2757247_i2304-dino-plugin-2-v4.job","0-8204841827243886683_2026-11-20T07-02-52-756113316Z_2757315_i2304-dino-export-grid-v4.job","0-8204841826981086319_2026-11-20T07-02-53-018913680Z_2757391_i2304-dino-grid-3a-v4.job","0-8204841826706861780_2026-11-20T07-02-53-293138219Z_2757458_i2304-dino-grid-3b-v4.job"]}}