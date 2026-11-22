#2388の4項目を実装し、`codex/parent-2384-issue-2388-primary` をpushしました。コミットは `7f35d62e9` です。[PR #2394](https://github.com/Motoki0705/tennis-lab/pull/2394)

- `locked (initializing)`、空index、Gitリンクのみの専用worktreeを再作成します。未知のファイルや別branchがある場合の拒否も確認しました。
- 汎用の公開失敗では、例外型と固定要約をprivate journalに保存します。明示的な再試行が必要な記録には、statusと公開CLIで `--retry` を案内します。
- 読めないjournalは、判明した親番号・名前空間のdigest・理由とともに残件に表示します。
- 両ログのPR HEADを最大3回再取得します。上限まで一致しない場合は、固定した公開データを保持して `pr_identity_changed` で停止します。

関連テスト454件と、コミット時のruff・mypyが通過しました。GitHubの応答遅延は実Gitとテスト用GitHub応答で検証しました。現HEADのCIは結果待ちで、実GitHubでの専用ログ公開はマージ後の確認に残ります。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2394,"head":"7f35d62e93df9bfbc10d9efc749030e019e8e82a","resume_on_success":false}}