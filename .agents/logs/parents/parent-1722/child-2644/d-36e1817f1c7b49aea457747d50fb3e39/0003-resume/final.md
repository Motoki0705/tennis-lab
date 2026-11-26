公開専用再開で欠けていた実装識別の照合を追加し、標準データセット公開でも同じ照合を行うようにしました。修正を `codex/parent-1722-issue-2644-primary` の `1b423fcdfe62986b55e1edea11d79b930b28ae2c` にpushし、実行証拠を更新しました。 [PR #2657](https://github.com/Motoki0705/tennis-lab/pull/2657)

commit だけが変わった場合は計算時の来歴を保持して再開し、ソースが変わった場合は公開前に拒否することを確認しました。旧 `scene-completion.json` の復旧手順をREADMEに追記しました。関連テスト139件とpre-commitが通過しました。

最終HEADでは配布版の固定ハッシュ、代表クリップのv3読込、#2613の入力照合を再確認しました。全件のチェックサム走査と前処理は初回HEADでの実行としてPR本文に記録しています。新HEADをCI監視に引き渡します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2657,"head":"1b423fcdfe62986b55e1edea11d79b930b28ae2c","resume_on_success":false}}