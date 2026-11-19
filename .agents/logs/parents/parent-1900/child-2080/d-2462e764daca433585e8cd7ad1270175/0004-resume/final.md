同じbranch `codex/unassigned-decision-publication` へ修正をpushしました。[PR #2087](https://github.com/Motoki0705/tennis-lab/pull/2087)

最終HEADは `8e42b0528f620db4352bb2461ea54577dc2ea5b5` です。

ignore対象を含む計画外の未追跡ファイルを検出し、内容を保持したまま再開を `worktree_dirty` で拒否します。commit済みworktreeの検査にも適用し、cleanupでは削除せず `retained_worktrees` に計上します。

追加した回帰テストは計6ケースです。

- `test_publication_resume_retains_ignored_outside_plan_files` — 4ケース。stage後・commit直後の中断から、保持・拒否・明示修復後の再開を確認。
- `test_publication_cleanup_retains_ignored_outside_plan_files` — 2ケース。merge確認後と再試行時の保持を確認。

親自身・担当なし親の両CLIで検証しました。release／handoff／supervisorを含む関係テストは計401ケース成功し、pre-commit・ruff・mypyも成功しています。

最新HEADのCIは初回実行でPythonを含む全14ジョブ成功です。worktreeはcleanで、起動した検証プロセスの残存もありません。[CI run 43675301905](https://github.com/Motoki0705/tennis-lab/actions/runs/43675301905)

未解決事項はありません。親側の差分確認・merge・実データ公開が残ります。