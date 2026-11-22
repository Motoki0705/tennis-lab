[PR #1793](https://github.com/Motoki0705/tennis-lab/pull/1793)を更新しました。branchは `codex/explain-html-service`、HEADは `34034817a3a1b07c9ed753a282c40ed18ccaa9f4` です。

undoは失敗確定前に公開revisionを照合し、公開済みなら成功として記録します。error付き旧journalも再起動時に修復します。照合不能は成功扱いしません。

点検でexportにも同型の問題を確認し、manifest・全公開bytesの照合を追加しました。編集commitには既存の照合があり、stageは非公開、session書込み失敗時は起動を中止します。

回帰テスト：

```sh
d=tests/integration/agents_ops/explain_html
.venv/bin/python -m pytest -n 0 \
 "$d/test_jobs.py::test_undo_publication_fault_reconciles_retries_and_restart" \
 "$d/test_jobs.py::test_recovery_reconciles_legacy_error_restore_journal" \
 "$d/test_revisions.py::test_snapshot_publication_fault_reports_only_verified_exports"
```

- 修正前：**4 failed / 3 passed、1.63秒**
- 修正後：**7 passed、1.55秒**
- 関連回帰：**200 passed、19.16秒**
- pre-commit・追加ruff/mypy：成功
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42593823840)：Python・Lint・CPU全10ジョブ・lane plan成功。初回の変更外Git pagerテストのSIGPIPEは、失敗ジョブのみ再実行して成功しました。

HTTP入口・状態名・token受渡し・CLI引数の変更はありません。作業treeはcleanです。統合レビュー・mergeを親へ返します。merge後main CIは未実施です。