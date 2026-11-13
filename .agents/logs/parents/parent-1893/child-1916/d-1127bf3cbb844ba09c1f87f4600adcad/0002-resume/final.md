[PR #1952](https://github.com/Motoki0705/tennis-lab/pull/1952) のmajor 2件を修正しました。`codex/main-git-sync`へ通常push済みです。追加commitは`dad7ca22f`、HEADは`abeb4f811`。worktreeはcleanです。

- HUP停止・停止確認・再起動確認をphaseとPIDで保存し、中断後も復旧記録を保持します。
- 手動運用CLIをlauncher必須にし、READMEと後続jobの接続手順を統一しました。

新規13ケース、最終重点回帰162件、pre-commit・ruff・mypyが成功。修正を外すと新テストが失敗することも確認しました。[対象HEADのCI #4069](https://github.com/Motoki0705/tennis-lab/actions/runs/43372998660) は`Python`を含む全14ジョブ成功です。

広めの回帰は1,331成功・1skip・1失敗。失敗は既存fakeの`os.pidfd_open`未提供で、mainでも再現済みです。

実ホスト検証は未実施です。親側には指摘解消確認・merge、#1917のlauncher接続、merge後mainのCI確認が残ります。