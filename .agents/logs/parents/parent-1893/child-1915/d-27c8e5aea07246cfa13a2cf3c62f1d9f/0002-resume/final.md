`spin ops` の6操作・対象表・停止後の残存照合を実装しました。[PR #1918](https://github.com/Motoki0705/tennis-lab/pull/1918) に `Closes #1915` を記載済みです。

- branch: `codex/spin-ops-control`
- commit: `34db76104`（push済み・worktree clean）
- 検証: 関連513件、pre-commit、Ruff、mypy成功
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43179178357): 全10分割・`Python` 成功

実ホストへの導入・有効化は未実施です。統合レビュー、merge、merge後mainのCI、実ホスト検証を親へ引き継ぎます。追加判断はありません。