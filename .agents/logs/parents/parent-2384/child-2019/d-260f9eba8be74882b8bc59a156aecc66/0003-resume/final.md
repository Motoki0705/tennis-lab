正式レビューの major 5 件を修正し、同じブランチへ push しました。修正コミットは `bb3fa5178`、新しい HEAD は `a4d43f38f` です。[PR #2399](https://github.com/Motoki0705/tennis-lab/pull/2399)

CLI は状態パスのシンボリックリンクを拒否し、レジストリのロックと読み書きを同じディレクトリに固定します。v2 索引の書き込み失敗後に旧状態を読み直して再試行できることと、dispatcher が私的なパスや内容を伏せて原因と復旧手順を出すことを確認しました。

関連テスト 639 件、main 取り込み後の回帰・ログ公開テスト 461 件、pre-commit の Ruff と mypy が通過しました。新しい HEAD の CI は未確認です。

実機の既定経路では `.local` と `.local/state` が `0775` のため、新しい検査が拒否します。実機反映前に書き込み権限の調整が必要です。v2 の複数ファイル置換中に `SIGKILL` された場合は、バックアップからの復旧が必要です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2399,"head":"a4d43f38fa7716b54f0bc6ab1e19062042bf2004","resume_on_success":false}}