指摘 `pr-2034-1` を修正し、[PR #2034](https://github.com/Motoki0705/tennis-lab/pull/2034)へ通常pushしました。消滅した親の観測を保持し、reparentした子も追跡します。

- branch: `codex/worktree-cleanup-host-fixes`
- commit: `6a91e1516`
- 内部境界テスト10ケース追加。故障注入で失敗し、復元後に成功することを確認。
- 関連278テスト、pre-commit、対象HEADのCI・`Python` checkが成功。

本番操作は未実施です。指摘解消確認・merge・実ホスト実行を親へ引き継ぎます。追加判断依頼はありません。