pr-2149-1を修正し、[PR #2149](https://github.com/Motoki0705/tennis-lab/pull/2149)へ通常pushしました。

- branch: `codex/review-resolve-base-merge-scope`
- commit: `ee85e3ada`
- 継承を「PR差分が変わらない場合だけ」に統一。complete-evidence・inspectにも適用しました。
- 関連738件・pre-commit成功。故障注入で4件失敗、復元後4件成功を確認しました。
- [CI #4288](https://github.com/Motoki0705/tennis-lab/actions/runs/43733003453)は`Python`を含む全14 job成功。作業treeはclean、最新mainとの競合もありません。

監査・検証結果はPRと子issueへ記録済みです。親による確認・resolve・mergeへ引き渡します。実resolve/gate、systemd・git-sync操作は行っていません。