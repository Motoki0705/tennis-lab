4件すべて修正し、[PR #1953](https://github.com/Motoki0705/tennis-lab/pull/1953)へ通常pushしました。

- branch: `codex/worktree-expiry-cleanup`
- commit: `f1a03dcc9`
- 関連回帰247件、運用回帰314件、最終専用51件（+13）が成功
- 修正外し5条件で回帰テストの失敗を確認
- pre-commit成功、対象HEADの[CI `Python`](https://github.com/Motoki0705/tennis-lab/actions/runs/43371351141)成功
- ローカル・リモート・PR HEAD一致、clean、競合なし

PR本文と子issueへ証拠を記録しました。競合回避のため、登録解除後の空管理ディレクトリは通常のGit prune/GCへ残します。

残作業は指定どおり、#1952 merge後の共通launcher接続です。実repoの削除・ホストsystemd操作は行っていません。