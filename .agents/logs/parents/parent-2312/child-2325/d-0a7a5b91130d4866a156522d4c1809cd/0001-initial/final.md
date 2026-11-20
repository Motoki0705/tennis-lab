#2325 を実装し、日本語の非ドラフト [PR #2340](https://github.com/Motoki0705/tennis-lab/pull/2340) を作成しました。ブランチは `codex/parent-2312-issue-2325-primary`、コミットは `abddf59c98a734d6874a6c9d325f842987a02540` です。

共有4層の `add`・`update`・`delete`・`batch` を専用CLIに統一しました。最新の `main` を一時 index に取り込み、生成文書と実差分を検査してから、PR を固定した head SHA でマージします。競合時は最大3回載せ直します。CI 除外と、自己修復・AutoDream 接続の運用文書も更新しました。

- 指定の pytest は **386件通過**。`validate`・`sync-index --check`・`audit`、Ruff、mypy、pre-commit も通過しました。
- 実リポジトリで dry-run を確認し、マージと拒否経路は実 Git と模擬 GitHub で検証しました。
- `.agents/ops/review/`、公開ログ、既存memoryの差分は0件です。全範囲の Ruff 検査で検出した既存テスト1行を整形し、そのファイルの33件も通過しました。

GitHub API は head SHA を固定します。`main` の ref は PUT 直前に照合しますが、照合とマージの間に `main` が進む可能性は残ります。実装PRの `Python` CI は親へ引き継ぎます。[GitHub REST API のマージ仕様](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2340,"head":"abddf59c98a734d6874a6c9d325f842987a02540","resume_on_success":false}}