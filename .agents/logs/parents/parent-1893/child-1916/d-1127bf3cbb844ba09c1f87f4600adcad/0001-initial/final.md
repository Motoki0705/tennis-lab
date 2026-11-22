[PR #1952](https://github.com/Motoki0705/tennis-lab/pull/1952) をレビュー可能にしました。

- branch: `codex/main-git-sync`[local path]`db30761e6`
- 固定SHAへの定期ff更新、ジョブ排他、OFF保持・中断復旧を実装。共通フックは独立commitに分離。
- ローカル **675 passed / 1 skipped**。ruff・mypy・pre-commit成功。
- [CI #4055 attempt 2](https://github.com/Motoki0705/tennis-lab/actions/runs/43357178890/attempts/2) は全10分割・`Python` 成功。初回の既存procfsテストのraceはPRに記録済み。
- validatorは未指定のため試行0／完了0。

親側には、#1917のlauncher接続、base載せ替え、統合レビュー・merge、実ホストとmerge後main CIの確認が残ります。