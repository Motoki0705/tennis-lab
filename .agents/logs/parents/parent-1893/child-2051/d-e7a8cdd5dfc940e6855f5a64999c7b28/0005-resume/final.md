[PR #2052](https://github.com/Motoki0705/tennis-lab/pull/2052) に、再起動後の個体照合と旧state移行を実装し、通常pushしました。

- branch `codex/worktree-cleanup-uncertain-scope`
- HEAD `b3e1bdd87bd0dfdb0bed498a1d3ec241a22e04d1`
- 関連346件、競合テスト計40件、故障注入・復元検証、pre-commit成功。
- 初回CIのmock干渉を修正し、最終HEADはPythonを含む全15 check成功。
- 本番コピーでは55件が移行・照合一致、消失済み29件を理由付きで終了。対象6パスは各1件、preflight 6件・エラー0でした。

対応はext4です。drvfsは識別不能として明示失敗にします。旧実体btime未保存分は、保存済み管理btimeとGit双方向登録を根拠に移行します。

統合レビュー・merge後の実ホスト削除確認が残ります。詳細は[担当issueの検証記録](https://github.com/Motoki0705/tennis-lab/issues/2051#issuecomment-6888011376)に記載しました。validatorは指定なし、試行0・完了0です。