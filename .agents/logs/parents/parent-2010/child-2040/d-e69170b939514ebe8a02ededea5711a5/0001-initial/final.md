[PR #2054](https://github.com/Motoki0705/tennis-lab/pull/2054) を作成しました。`Closes #2040` 付きです。

- branch: `codex/gh-http-client`
- HEAD: `ff1397e65540f0ae6efd152bfbdcfa40d24cc97b`
- HTTPS接続を再利用し、repo省略時はgit remoteから解決。既存の鮮度・ロック・再試行契約を維持しました。
- delegateのAPI件数は **59→53回**。実GitHubの3 GETは **3.212→1.553秒**。計測表と制約をPRに記載しました。
- 関連365件、追加smoke、pre-commit成功。**対象HEADのPython CIも成功**しています。初回のannotation権限エラーは、コード変更なしの再実行1回で成功しました。

全体実行の既存失敗5件は変更前mainでも再現し、PRに記録済みです。新規GraphQLバッチ化・並列化は未実施。validatorは指定なし、試行／完了0／0です。

統合レビュー・merge・merge後mainのCI確認を親へ返します。