[PR #2087](https://github.com/Motoki0705/tennis-lab/pull/2087) に修正をpushしました。最終HEADは `1074059b9df763f2439d4269f8c529e485e2c524` です。

- **pr-2087-1**：merged batchの未完了処理を、新しい要求の作成より先に照合します。要求の公開直後に停止しても、凍結した要求IDを再利用し、追加PRを作りません。追加テスト：`test_unassigned_publication_resumes_frozen_actor_after_partial_confirmation`。
- **pr-2087-2**：git commit直前とcommitted状態の保存前に担当を再照合します。中断したworktreeは凍結した計画との一致を検証して再開し、未知の変更は拒否します。`test_unassigned_publication_rechecks_before_external_writes` にplan・staging・検証後などのケースを追加しました。
- **同種の箇所**：batch保存、journal追記、PR作成・mergeの送信直前、送信失敗後の保存にも再照合を追加しました。追加テストには `test_publication_rechecks_owner_after_saving_dispatch_intent`、`test_failed_dispatch_does_not_save_bookkeeping_after_owner_change`、`test_publication_journal_rechecks_owner_after_validation` があります。

関係テストは重複を除く **621ケース成功**、今回の追加は21ケースです。pytestは2並列で分割実行し、pre-commit・ruff・変更Pythonファイルのmypyも成功しました。起動した検証プロセスの残存はありません。

[最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43665889199) は初回実行でPythonを含む全14ジョブ成功です。worktreeはclean、PRもmerge可能です。

未解決事項はありません。実stateへの書込み・公開は行っておらず、統合レビュー・merge・実データ公開は親側に残ります。