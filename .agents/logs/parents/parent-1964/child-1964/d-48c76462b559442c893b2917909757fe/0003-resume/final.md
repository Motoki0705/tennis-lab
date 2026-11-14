[PR #1971](https://github.com/Motoki0705/tennis-lab/pull/1971) にorigin/main（`f192c35a3`）をmergeしてpushしました。

branch: `codex/issue-proposal-label`  
merge commit／HEAD: `6826d80fc0fd3339df1dd2c394909a4e0cc3cd59`

- `test_ops_control.py` の衝突は、両方のunitコピー・期待対象・専用テストを保持して解消しました。`proposal-expiry` の単発実行ケースも追加しました。
- 自動mergeされたops_controlのREADME・targets、parentのREADME・sessionを確認し、mainのruntime来歴保存と既存の承認gateを保全しました。
- ops_controlへの追加登録は不要です。既存の `proposal-expiry` timer/service登録と、mainの `worktree-cleanup` が共存しています。

検証は先行61ケース成功。広域回帰は **728成功・2失敗**でした。同じ停止来歴エラーをorigin/main単体の並列実行でも再現しています（121成功・1失敗）。対象ファイルの小規模再実行はmerge側・main側とも3ケース成功。間欠失敗は未修正としてPRへ記録しました。取込み63ファイルへのpre-commitも成功しています。

[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43575389746) は **Pythonを含む全14ジョブ成功**。PRは `MERGEABLE / CLEAN`、worktreeもcleanです。本文の `Closes #1964` を維持しました。