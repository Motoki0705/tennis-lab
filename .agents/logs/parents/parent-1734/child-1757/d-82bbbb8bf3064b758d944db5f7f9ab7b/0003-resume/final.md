**GPUの定期照会が計測停止に関与する仮説が強まりました。** 同じ `clip_027/cam0` をDINO fp16・batch size 1で実行しました。

| 条件 | 結果 | 所要時間 |
|---|---|---:|
| A・GPU照会なし | 235/235フレーム完走 | 133.19秒 |
| B・`nvidia-smi` を1秒ごとに照会 | 52フレーム完了。53番目の推論中に照会がタイムアウトし、15:15:45 JSTに監視が停止 | 84.04秒 |

Bでは停止判定の0.17秒前にもDINOのheartbeatが更新されていました。WSLでは15:15:44 JSTに `dxgvmb_send_sync_msg: wait_for_completion failed` が1件あり、Windowsの `nvlddmkm` は両条件とも取得範囲内で0件でした。各条件1回の比較で、過去のGPU停止の原因は未特定です。 [実験記録](https://github.com/Motoki0705/tennis-lab/blob/3c627cb0ae884415aef00aa99edacfd6e6aa619c/knowledge/nodes/tennis_scene/000073-run-i1757-dino-gpu-health.md)

共有queueのjob `0-8204931363351882979_2026-11-19T06-10-36-648117020Z_1140385_i1757-dino-gpu-health` は `failed`、終了コード1です。再試行は0回です。 [queueログ]([local path]) [条件別結果](https://github.com/Motoki0705/tennis-lab/blob/3c627cb0ae884415aef00aa99edacfd6e6aa619c/outputs/tennis_scene/evaluate/dino-gpu-health-20261119/results.json) [WSLログ](https://github.com/Motoki0705/tennis-lab/blob/3c627cb0ae884415aef00aa99edacfd6e6aa619c/outputs/tennis_scene/evaluate/dino-gpu-health-20261119/B/kernel-dxg.log) [Windowsログ](https://github.com/Motoki0705/tennis-lab/blob/3c627cb0ae884415aef00aa99edacfd6e6aa619c/outputs/tennis_scene/evaluate/dino-gpu-health-20261119/B/windows-nvlddmkm.log)

実測commitは `429065101`、`codex/video-batch-benchmark` の公開HEADは `3c627cb0a` です。実測後にmainを取り込み、競合したmypy設定は両方残しました。既存98テスト、pre-commit、Knowledge libraryは成功しました。CIはmainでも再現するBall Detectionのforward契約テスト2件で失敗し、結果をDraft PRに記録しました。逐次本計測と一括処理の速度・品質比較は未実施です。 [Draft PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273) [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/44471514165) [mainの同一エラー](https://github.com/Motoki0705/tennis-lab/actions/runs/44468675000/job/135377618766)