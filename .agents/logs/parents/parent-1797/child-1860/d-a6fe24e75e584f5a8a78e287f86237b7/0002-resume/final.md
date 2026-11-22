[PR #1867](https://github.com/Motoki0705/tennis-lab/pull/1867)を作成しました。実装済みですが、CI成功の受入は未完了です。

- Branch: `codex/review-stage-recovery`
- Commit: `24231e6f8`
- 関連354ケース、pre-commit成功。故障注入2件の検出・復元も確認済み。
- CIは8レーン成功。0・5がapt取得で再度タイムアウトし、[Pythonはfailure](https://github.com/Motoki0705/tennis-lab/actions/runs/42804039594/job/129801641561)です。

CI基盤 #1857 の復旧後の再検証と、親の統合レビュー・merge・merge後main CI確認が残っています。