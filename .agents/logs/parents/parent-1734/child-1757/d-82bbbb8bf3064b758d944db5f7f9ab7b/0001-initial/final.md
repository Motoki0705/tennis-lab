[Draft PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273) に3クリップの計測条件、再現手順、15:00 JST締切の状況を記録しました。ブランチは `codex/video-batch-benchmark`、HEADは `d709f3ed97f69b8f7ca8e330ebf4fc7063f24002` です。

**逐次の本計測は未開始です。** 13:32 JSTにGPU probeを開始し、`clip_027/cam0` のBall検出結果の保存とDINO fp16の実推論を確認しました。`nvidia-smi` が10秒以内に応答せず、13:36 JSTに監視がprobeを停止しました。再試行は行わず、14:19 JSTにqueueの `failed` とログの `exit_code=1` を確認しました。

- Job ID `0-8204937387912385687_2026-11-19T04-30-12-087614312Z_640052_i1757-meiji-serial`。終了状態は共有queueの `list` とログで照合できます。 [queueログ]([local path])
- 実行前に固定した[許容誤差](https://github.com/Motoki0705/tennis-lab/blob/d709f3ed97f69b8f7ca8e330ebf4fc7063f24002/knowledge/runs/run-i1757-video-batch-meiji/tolerances.json)と、実行commit・停止時刻を含む[probe記録](https://github.com/Motoki0705/tennis-lab/blob/d709f3ed97f69b8f7ca8e330ebf4fc7063f24002/knowledge/runs/run-i1757-video-batch-meiji/probe-summary.json)を保存しました。
- 公開HEADのPython・Lint・Knowledge libraryは成功しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44461931772) [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44461931732)

残るのは、逐次の初回・warm計測、モデル再利用のみ、#1755の横断batch、#1756のembedding再利用です。player証拠の方向一致、scene再読込と品質差分、差分動画、実GPUでの中断・再開とcache混在も未確認です。