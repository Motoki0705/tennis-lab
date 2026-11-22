#2358 を実装し、指定ブランチ `codex/parent-2384-issue-2358-primary` のコミット `43fbd80c1` を push しました。 [PR #2397](https://github.com/Motoki0705/tennis-lab/pull/2397)

- `merge-continue --run-id RUN --retry-review ROUND --instruction-file FILE` で、前回の終了・回収、対象 PR と HEAD、実行権限を照合し、最大2回まで再試行できます。終了を確認できる記録が不足する旧 run は保留します。
- 完了済みのレビュー出力を AI 再起動ゼロで回収・公開できることを確認しました。横断指摘は対象 PR に紐付け、対象外 PR は参照として保存・表示します。

関連 pytest 529 件、ruff、mypy、pre-commit が通過しました。503 相当の障害と CI 失敗時のマージ抑止はテストで確認しました。対象 HEAD の CI はキューに入り、結果確認は親へ引き継ぎます。 [CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/45024925531)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2397,"head":"43fbd80c11a8ccc75387e28df65013d330f761c9","resume_on_success":false}}