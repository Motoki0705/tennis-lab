GPU受入はハング疑いで中断しました。[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)はDraftを維持しています。

- **11/18 01:43:07.588 JST**、基準mainで `worker_heartbeat_timeout`（180秒）を検出。最後の観測はDINOの `DeformableTransformer.forward`、`deformable_transformer.py:289` でした。
- NVMLは直前まで応答していました。明示的なCUDA例外文・停止stackは取得できず、原因は未確定です。
- mainを終了し、再投入は0回。runtime側は未起動で、完成出力比較・読込回数・peak VRAMの受入確認は未完了です。[エラー全文・実行command](https://github.com/Motoki0705/tennis-lab/blob/1bef628a9/knowledge/runs/run-i1751-runtime-gpu-a4/failure-report.json)と[knowledge記録](https://github.com/Motoki0705/tennis-lab/blob/1bef628a9/knowledge/nodes/tennis_scene/000051-run-i1751-runtime-gpu-a4.md)を保存しました。
- branchは `codex/scene-inference-runtime`、HEADは `1bef628a9`。main `0daeba0c5` まで統合済みです。
- CPU関連354件、設定回帰65件、監視smoke 9項目、ruff/mypyは成功。最新HEADの[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44131465350)は全成功、[CPU CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44131464826)は実行中です。

原因確認とGPU再開の判断を親へ返します。共有queueの受付停止flagは設定済みですが、別担当の `video-batch-runtime-parity-reboot-a2` は実行中で、共有worker自体は稼働しています。