正式レビューのmajor 3件を修正し、`codex/parent-2384-issue-2388-primary` にコミット `05d7ee127` をpushしました。`Closes #2388` を維持し、修正と検証の結果を追記しました。[PR #2394](https://github.com/Motoki0705/tennis-lab/pull/2394)

- worktree復旧では旧checkoutとGit登録を非公開の退避先に移して再検査します。直前に追加されたファイルや `manual` lock を保持して公開を拒否することと、退避途中から再開できることを確認しました。復旧後も旧個体は退避先に保持します。
- 読めないjournalの識別情報は、判明した親番号と相対パスのSHA-256に限定しました。未検証の32桁hex文字列が一括照合やstatusに出ないことを確認しました。
- 親・issue・委任キー・turnの各ディレクトリ位置に通常ファイルがある場合は `journal_unavailable` として記録し、終了checkpointに名前空間と `remaining_complete=false` を残します。

関連テスト462件と、コミット時のruff・mypyが通過しました。新HEADのCIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2394,"head":"05d7ee127fb36f0a4e05b026aa9f347f4b574de3","resume_on_success":false}}