`main` を取り込み、2ファイルの競合を解消しました。同じブランチ `codex/parent-1722-issue-2644-primary` に `f05dd9f63b7867710fd57d08a7d382b2489edc74` をpushし、PR本文の実行証拠を更新しました。GitHubでも競合解消を確認しました。 [PR #2657](https://github.com/Motoki0705/tennis-lab/pull/2657)

関連テスト141件とpre-commitが通過しました。ソース識別は全16設定で `main` と一致し、個別にソースを変えた160条件では識別が変わりました。配布版の固定ハッシュ、代表クリップのv3読込、#2613の入力照合も新HEADで再確認しました。全件検証は初回HEADの実行としてPR本文に記録しています。

新HEADのCIは開始待ちです。 [CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/45827988339)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2657,"head":"f05dd9f63b7867710fd57d08a7d382b2489edc74","resume_on_success":false}}