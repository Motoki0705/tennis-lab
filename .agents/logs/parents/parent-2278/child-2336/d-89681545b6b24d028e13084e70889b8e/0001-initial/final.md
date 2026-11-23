#2336のバッチ掃引を実装し、52ケースを対象とする3件のGPU計測ジョブを共有training queueに登録しました。ブランチは `codex/parent-2278-issue-2336-primary`、コミットは `af205aa9c976543eb914efe294641ddf2c19f4a6` です。

固定入力・重み・DINO CUDA拡張のSHA-256を先行runと照合しました。CPUテストは80件成功し、Ruffも通過しました。ViTPose-Hは反転込みでTensorRT内部の入力を2Bとし、throughputは元のcrop数で集計します。DINOは750×1333でGridSample版とプラグイン版を測ります。 [再現手順](https://github.com/Motoki0705/tennis-lab/blob/af205aa9c976543eb914efe294641ddf2c19f4a6/knowledge/runs/run-i2336-vision-batch-sweep/README.md)

受入条件の実GPU証拠はジョブ結果待ちです。結果を受け取ったら速度・VRAM・nsys・精度と、TensorRTの範囲外拒否・層精度を確認します。その後、代表バッチのncuを採取し、集計とPR作成に進みます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204694588545189554_2026-11-21T23-56-51-454810445Z_4147028_i2336-vitpose-timing.job","0-8204694588289761695_2026-11-21T23-56-51-710238304Z_4147065_i2336-hmr2-timing.job","0-8204694588023825357_2026-11-21T23-56-51-976174642Z_4147159_i2336-dino-timing.job"]}}