main `e1016bb8e` へのrebaseと横断majorの修正を完了し、[PR #1953](https://github.com/Motoki0705/tennis-lab/pull/1953)を更新しました。指定leaseでのforce pushは1回だけ使用しました。

- branch: `codex/worktree-expiry-cleanup`
- HEAD: `b781b9bc2`
- service・手動spinを共通launcherへ接続し、直接module実行を拒否
- 双方向の排他、resume記録・手動OFFの保持を確認
- 関連431件、最終CLI確認18件が成功。追加8ケース、修正外し5条件も検証
- PR全差分のpre-commit成功、[対象HEADのCI `Python`](https://github.com/Motoki0705/tennis-lab/actions/runs/43386056268)成功
- HEAD一致・clean・競合なし。validatorは未指定、試行0／完了0

実repo削除・ホストsystemd操作は未実施です。実ホスト検証、統合レビュー・merge、merge後main CI確認を親へ引き継ぎます。