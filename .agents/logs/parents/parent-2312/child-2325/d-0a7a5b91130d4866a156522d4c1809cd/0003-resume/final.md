正式レビューの5件をコミット `88ccc95432fbe1c6896e81bb98992692b989817b` にまとめ、同じブランチ `codex/parent-2312-issue-2325-primary` へ push しました。PR本文を更新し、回帰テストの結果をコメントに記録しました。[PR #2340](https://github.com/Motoki0705/tennis-lab/pull/2340) · [検証コメント](https://github.com/Motoki0705/tennis-lab/pull/2340#issuecomment-7045388282)

1. **マージ時の競合**。GitHubの `updateRefs` で `main` とPRブランチのSHAを同じトランザクションで照合します。最終照合後に引用先の見出しが消えた場合、更新が拒否され、載せ直し時の引用検査で停止することを確認しました。[GitHubの仕様](https://docs.github.com/en/graphql/reference/git#updaterefs)
2. **古い更新・削除計画**。`expected_sha256` を追加し、再検証済みのエントリに対する古い計画を初回適用時と載せ直し時の両方で拒否することを4ケースで確認しました。
3. **CI除外の境界**。`paths-ignore` は確実に一致する名前に絞り、ハイフンを含む名前は `scope` がCLIと同じ規則で判定します。両トリガーの28ケースで、不正名、ネスト、非Markdown、`ops-report`、コード混在が通常CIの対象になることを確認しました。[Actionsのパターン仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#filter-pattern-cheat-sheet)
4. **生成文書の修復**。`repair-index` と `publish_index_repair` を追加し、生成文書だけの差分を公開できることと、同期済みの場合に公開前に拒否することを確認しました。
5. **SIGTERM後の資源回収**。Gitオブジェクトからコミットを作る方式に変更しました。実プロセスを中断し、worktree登録、一時ファイル、子Gitプロセスが残らないことを確認しました。

関連テストは **589件、204.02秒**で通過しました。`validate`、`sync-index --check`、`audit`、実リポジトリでの dry-run、Ruff、mypy、pre-commit も通過し、`.agents/ops/review/` の差分は0件です。マージ動作は実Gitと模擬GitHub APIで検証しました。新HEADの `Python` CI は親へ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2340,"head":"88ccc95432fbe1c6896e81bb98992692b989817b","resume_on_success":false}}