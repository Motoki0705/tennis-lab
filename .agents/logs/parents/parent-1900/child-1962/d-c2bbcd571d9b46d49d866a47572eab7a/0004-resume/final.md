[PR #1973](https://github.com/Motoki0705/tennis-lab/pull/1973) を更新・通常pushしました。

- branch: `codex/log-turn-traceability`
- 最終HEAD: `f5df79f67b39926a64bf073d32a5c561240cebdc`
- origin/main `f94c99c7` を通常merge。手動解消が必要な競合はありませんでした。

release、handoff pending／accepted、supervisor終了を接続しました。順序は「担当変更保存 → worker公開 → 終了観測捕捉 → 判断ログ公開 → 最終checkpoint保存」です。捕捉・公開失敗でも終了を保持し、公開処理による循環を防ぎます。公開する観測は判断ログ公開直前、最終private checkpointは公開後の状態です。

追加テストは計8ケースです。

- `test_exit_after_publication_links_acceptance_and_end_without_publication_loop`
- `test_interrupted_exit_reconciliation_keeps_remaining_actor_and_resume`
- `test_supervisor_exit_publishes_typed_checkpoint_and_explicit_acceptance`
- `test_exit_keeps_unknown_inventory_or_capture_failure_without_veto`
- `test_lookalike_result_payload_and_summary_do_not_supply_acceptance`

関連実行は1010成功・旧期待値4件失敗を修正し、修正後の対象142件が成功。pre-commit成功。[最新HEADのCI #4188](https://github.com/Motoki0705/tennis-lab/actions/runs/43638869434) は全10 CPU shard・Lint・system dependency・集約Python成功です。初回のプロセス由来確認1件は再実行で成功しましたが、並行時の失敗原因は未確定です。

過去事例は親1796／1864の同じstateで `historical.capture(decision_journal(...))` を実行し、`publish-decision-log --dry-run` で確認後に公開します。#1816ターン2・3は1796に含まれます。具体コマンドをPR本文に記載しました。

実stateへの取り込み・公開とmerge後のmain確認は未実施です。起動プロセスの残存はなく、作業ツリーはクリーンです。