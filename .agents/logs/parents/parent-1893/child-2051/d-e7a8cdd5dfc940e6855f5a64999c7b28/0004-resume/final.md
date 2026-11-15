[PR #2052](https://github.com/Motoki0705/tennis-lab/pull/2052) を新方針で更新し、通常push済みです。

- branch `codex/worktree-cleanup-uncertain-scope`
- commit `d543aabd72edd909089b0eb1f19d7a6f4640682a`
- 関連326件、競合テスト各20回・計40件、故障注入後の復元検証、pre-commit成功。
- 対象HEADのCIはPythonを含む全15 check成功。
- 実ホスト6件は参照0・関連する確認不能0。関連未観測の確認不能3 PIDは診断扱いです。見えない参照の検出限界はREADMEに記載しました。

**親の判断が必要です。** WSL再起動後、保存device `2112` と現在の `2096` が相違し、旧intentの個体照合は6件とも失敗します。公開dry-runでは同じ6パスが計12件に分かれました。再開時の個体照合の扱いを親へ返します。

統合レビュー・merge・実ホストでの削除確認が残ります。証拠は[担当issueの記録](https://github.com/Motoki0705/tennis-lab/issues/2051#issuecomment-6887035560)にまとめました。validatorは指定なし、試行0・完了0です。