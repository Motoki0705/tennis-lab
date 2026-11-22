**条件2は本計測に入りました。** probeは19:05:42 JSTに正常終了し、本計測のheartbeatが19:06:06から19:07:27へ更新されたことを確認しました。2ジョブとも公開HEAD `795d63df1682d0d86f769d584dfbfa263419ad0d` で、親が起動した共有workerが管理しています。 [Draft PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)

| 条件 | queue job ID | 確認時の状態 |
|---|---|---|
| 2・モデル再利用 | `0-8204917572928512634_2026-11-19T10-00-27-071487365Z_1772136_i1757-reuse-heartbeat-1` | 本計測中 |
| CLIP embedding再利用 | `0-8204917572671425689_2026-11-19T10-00-27-328574310Z_1772178_i1757-embedding-reuse-heartbeat-1` | all枠待ち |

終了は共有queueの `done` と各runの `measurement/terminal.json` の `status: complete` を照合します。queueログは共有repoの `.training_queue/logs/<job-id>.log` です。 [条件2の本計測ログ](https://github.com/Motoki0705/tennis-lab/blob/795d63df1682d0d86f769d584dfbfa263419ad0d/outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/reuse-heartbeat-1/measurement/worker.log)

条件1の逐次計測は初回2,118.01秒（起動込み2,131.75秒）、warm 2,031.50秒・2,005.06秒でした。固定許容誤差で全3組を比較し、各組141 nodeの保存内容とscene配列が完全一致しました。9件すべて再読込でき、court_sideは全件 `[false,false,true,true]` でした。 [逐次の計測記録](https://github.com/Motoki0705/tennis-lab/blob/795d63df1682d0d86f769d584dfbfa263419ad0d/knowledge/runs/run-i1757-video-batch-meiji/serial-heartbeat-2.md)

mergeの競合はmainの実装を採用し、knowledgeノードを75・76に振り直しました。CPUテスト119件、pre-commit、Lint、Knowledge libraryは成功し、Python CIは実行中です。条件3は #2272 のmerge待ちです。逐次と新条件は実行commitが異なり、段階別時間には #2227 のX3D計測境界の変更が含まれます。 [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/44501370567) [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44501370640)