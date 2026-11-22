[PR #1984](https://github.com/Motoki0705/tennis-lab/pull/1984) を更新し、通常push済みです。

branch: `codex/log-publication-completion`  
最終HEAD: `52696fea83f99a43bbf89dc325db6b15f6563aa1`

4件とも再現テストの失敗を確認してから修正しました。

- **pr-1984-1**: journalごとの公開直前に世代を照合し、引継ぎ後の巡回を停止。`test_handoff_during_first_worker_merge_stops_later_journals` の4経路で検証。照合を外すと4件失敗。
- **pr-1984-2**: 未merge batchと捕捉envelopeのID・内容を照合し、不足を完了扱いにしない。`test_pending_batch_requires_its_captured_prepared_payloads` で検証。照合を外すと4件失敗し、IDだけの照合でも内容不一致の1件が失敗。
- **pr-1984-3**: 入れ子の公開失敗を外側へ伝播。`test_existing_journal_merge_refusal_reaches_orphan_cli_exit` で403時の終了値2を確認。修正を外すと終了値0となり失敗。
- **pr-1984-4**: `runtime_errors` の欠測を拒否。`test_missing_runtime_errors_is_not_an_explicit_empty_array` と `test_worker_receipt_uses_the_same_error_classification[missing_errors]` で検証。修正を外すと3件失敗、対照2件は成功。

進行中にmainとの競合が生じたため、通常mergeでCLIとstorageテストを統合しました。mainの待機・transaction契約を保持し、cache条件の食い違いも補正しています。

関連90ファイル **3,585件すべて成功**、pre-commit・commit時hook成功。[最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43614837092) はPythonを含む14jobすべて成功しました（attempt 1）。PRは競合なし、worktree差分・残存プロセスともに0です。

4指摘の未解決点はありません。以前記録された補助mypyの7件は今回未再検証です。統合レビュー・merge・実データの一括回収は親担当として残しています。