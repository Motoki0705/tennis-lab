[PR #1984](https://github.com/Motoki0705/tennis-lab/pull/1984) を修正し、通常push・検証欄の更新を完了しました。

- branch: `codex/log-publication-completion`
- 最終HEAD: `c9d169cce79c040939869511505076a4d68ebbdd`

公開開始前の照合を共通化しました。明示公開後の続行も最初の世代を維持し、世代を取得できない場合は公開を停止します。

| 公開入口 | 照合箇所 |
|---|---|
| 明示 `publish-worker-log`（`--turn`省略含む） | `_command` の各journal直前＋`service.publish` |
| 次の書込みCLI・release・handoff | `Continuation.sweep` → `continue_workers` → `publish` |
| supervisorの親終了処理 | `exit_publication_owner` → `finish_supervised` |
| `--all-parents` | `bulk_command` の固定した担当世代 → 通常続行 |
| orphan採用・元journalあり | 採用処理 → 世代照合付き通常続行 |
| orphan採用・元journalなし | 各PR・検査後、元turnのlock取得前 |
| `run-worker`完了 | 予約時の担当世代 → `finish_manual` → `publish` |
| supervisorの子完了 | registry内で固定した担当世代 → lock外の`publish` |

追加・拡張した回帰テストと、修正を外した結果です。

- `test_handoff_during_first_worker_merge_stops_later_journals`：明示入口は基準commitで再現。明示照合を外すと1件失敗。
- `test_manual_completion_preserves_worker_exit_after_publication_handoff`：上記入口群と合わせ、共通ガードを外すと8経路・9件失敗。
- `test_publication_admission_rechecks_its_frozen_owner`：世代比較を外すと1件失敗、対照2件成功。
- `test_decision_publication_postlude_keeps_its_original_generation`：世代固定を外すと明示公開側と合わせ2件失敗。初期世代欠測時の停止を外すと1件失敗。

関連テストは重複を除き593件成功。最終変更後の142件と、修正復元後の回帰14件も成功しました。pre-commit成功、[最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43629610392) はPythonを含む14ジョブすべて成功（attempt 1）です。

worktreeはクリーン、残存プロセス0、PRはmergeableです。今回の指摘に未解決点はありません。統合レビュー・merge・実データの一括回収は親担当に残ります。