正式レビューの major 2 件を修正し、`codex/parent-2384-issue-2358-primary` に push しました。`Closes #2358` を維持し、修正コミットと検証結果を PR 本文に追記しました。 [PR #2397](https://github.com/Motoki0705/tennis-lab/pull/2397)

- `7a3aae7b4` で、worktree 作成後に GitHub の HEAD と権限を再照合し、起動意図の保存からランナー起動までを権限変更と直列化しました。worktree 作成後の HEAD 変更・権限取消しでは、追加起動はともにゼロでした。
- `e979711ea` で、最初の結果保存前の中断から、AI 未起動を確認できた場合に再試行できるようにしました。実ランナーで復旧を再現し、AI 起動痕跡がある場合は追加起動ゼロで拒否することを確認しました。

関連 pytest 606 件、ruff、mypy、両コミットの pre-commit が通過しました。修正 HEAD の CI は進行中で、結果確認は親へ引き継ぎます。 [CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/45029769963)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2397,"head":"e979711eac2fcdadbcb8bb5b27dd5631a261f802","resume_on_success":false}}