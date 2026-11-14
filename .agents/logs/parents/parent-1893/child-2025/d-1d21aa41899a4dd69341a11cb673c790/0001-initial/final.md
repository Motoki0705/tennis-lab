[PR #2034](https://github.com/Motoki0705/tennis-lab/pull/2034) を作成しました。

- branch: `codex/worktree-cleanup-host-fixes`
- commit: `e94750302`
- 関連268テスト、新規20ケース、pre-commit成功。CIの必須 `Python` checkも再実行後に成功しました。
- 実ホストの診断用dry-runは94件選択、preflightエラー0。本番stateの変更・削除・unit操作は未実施です。

残作業は親による統合レビュー・merge、実ホストでの単発削除とunit更新です。追加の判断依頼はありません。