**本番a4はD:容量ガードで停止しています。** 最終確認の空きはD: **2.536GiB**、RAM available **16.538GiB**。容量回復前の再起動は行っていません。

- 検証済みT2・1,000更新から完全復元し、compile後10更新を確認。その後1,313まで有限・連続でした。最新の検証済み保存点は1,000で、未保存の313更新は再開に採用しません。
- RAMは更新前6GiB未満で無期限待機、2GiB未満で緊急停止へ変更。親子RSSを記録します。36テスト成功。VRAM最大 allocated **10.80GB／reserved 12.76GB**。
- [PR #1955](https://github.com/Motoki0705/tennis-lab/pull/1955)。実装commit `15cdc684f` と報告commit `a1cdadc57` のCI Pythonは成功。最新の停止記録 `130cce699` はKnowledge CI成功、**Python CIは進行中・未確認**です。
- unit `tennis-ball-arch-desktop-production-20261113-a4.service`、queue末尾 `3338941_ball-arch-desktop-production-a4` はともにfailed。出力rootは `outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261113-a4/`。
- 待機ログは同rootの `production/T2-q00000-q06000/ram-waits.jsonl`。正確なqueue ID、terminal event、未使用attempt 5での復旧手順は[handoff](https://github.com/Motoki0705/tennis-lab/blob/bcb21726223caa38f46916a321852e677df1811e/knowledge/runs/run-ball-arch-desktop-20261113/monitoring-handoff.json)に記録しました。

停止証拠と訂正済み引継ぎ資料73ファイルはDrive保存・独立readback済みです。残件はD:容量回復と再開判断です。定期follow-upは登録していません。