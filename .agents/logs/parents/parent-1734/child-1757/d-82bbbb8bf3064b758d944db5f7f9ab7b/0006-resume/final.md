**条件3は本計測中です。** probeは22:38:20 JSTに正常終了し、本計測のheartbeatが22:38:48から22:40:11へ更新されました。公開HEADは `fb6255a2853a6992e3d96287d7b989b681c2e336` です。 [Draft PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)

- job ID `0-8204904851837375583_2026-11-19T13-32-28-162624416Z_2750129_i1757-batched-heartbeat-1`
- #2272の既定値はCourt 2、DINO 2、ViTPose 4（flip時8）、CLIP 16、HMR2 4、Ball 1です。作業量上限は3 GiB、最大4 source・32 pending item、共有store cacheは0 byteです。
- 共有queueの `done` と `measurement/terminal.json` の `status: complete` で完走を確認できます。 [本計測ログ](https://github.com/Motoki0705/tennis-lab/blob/fb6255a2853a6992e3d96287d7b989b681c2e336/outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/batched-heartbeat-1/measurement/worker.log) [queueログ]([local path])

**条件2とCLIP再利用の結果を記録しました。** warm中央値はモデル再利用1,802.76秒、CLIP再利用1,819.36秒でした。CLIPは各回816/816件に命中しましたが、SHA-256を作る外観特徴段が毎回14.84〜16.02秒長くなり、全体の中央値も16.60秒増えました。両条件のGPUピークはallocated 4.185 GB、reserved 4.461 GBで、RSSの最大はそれぞれ9.442 GB、9.484 GBでした。 [段階別の計測記録](https://github.com/Motoki0705/tennis-lab/blob/fb6255a2853a6992e3d96287d7b989b681c2e336/knowledge/runs/run-i1757-video-batch-meiji/model-reuse-comparison.md)

18件のsceneを再読込し、court_sideは全件で参照と一致しました。同じHEADの両条件では最終scene配列が完全一致しました。旧逐次との比較ではボール出力が許容差を超え、027と037の計9フレームで3D有効性が変わったため、品質受入は未達です。差はボール検出のheatmap・候補から始まり、実行commitとモデル寿命の影響は未分離です。

CPUテスト87件、pre-commit、Lint、Knowledge libraryは成功しました。対象HEADのPython CIは実行中です。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44534332899) [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44534332966)