[PR #1942](https://github.com/Motoki0705/tennis-lab/pull/1942)を作成しました。未merge、worktreeはcleanです。

- branch: `codex/issue-standalone-label`
- HEAD: `1ea4ba8b1d3022eaedb08fa18d77b7dde69b4456`
- 19ファイル変更。ラベル定義・テンプレート、単独起票、排他・整合検査、親子移行・復旧、検索規約とテストを実装しました。
- 条件1の同期dry-runは作成1・更新0・削除0。条件2〜4・6は、role別ラベル、保存draft互換、矛盾検出、移行の再実行、下流回帰で確認し、PR本文に対応付けました。

ローカル589件成功（285.29秒）。初回CIで検出した子関係キャッシュの退行を修正し、修正後の43件・自動起票76件・番号prefix回帰1件も成功しました。コミット時のpre-commitも通過しています。

最新HEADの[CI `Python`は成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43343266621/job/131622308879)。全10 CPU分割とLintも成功しました。初回CIの失敗と対応はPRに記録済みです。

残作業は親の統合レビュー・merge、GitHubラベル反映・再同期、指定11件の再確認と限定移行、実検索の照合です。追加の広域mypyで出た既存11件の診断はmainと同一でした。validatorは未指定のため0回。判断依頼はありません。