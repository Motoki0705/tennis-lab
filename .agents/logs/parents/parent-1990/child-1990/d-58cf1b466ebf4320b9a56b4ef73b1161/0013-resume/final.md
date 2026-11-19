[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023)を更新しました。`codex/storage-tiering` / `be34c7b5f`、push済み・worktree cleanです。

- 指摘は修正前の5件失敗で再現。ckpt直下ファイルと通常ディレクトリに、永続した公開計画と[local path]`recover`を追加しました。旧manifest・payloadを履歴へ保持し、通常の不一致拒否は維持しています。
- 関連テスト678件成功（純増17件、58.53秒）。故障注入18件の失敗、復元後の成功、pre-commit・明示ruff/mypyを確認しました。
- 対象HEADの[CIはPythonを含む全14ジョブ成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43667725042)。実mount 10件はCIでは権限制約でskip、ローカルでは全件成功です。
- 実機の読取専用QAはunknown 0、atime・metadata・symlink不変。README 3言語とPR本文を更新し、`Closes #1990`を維持しました。

配置切替・実機recover・実削除・unit・sudo操作は未実施です。旧形式の未完了journalは手動確認まで停止します。validatorは未指定、実施0回です。