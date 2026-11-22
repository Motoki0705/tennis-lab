#2311 を実装し、`codex/parent-2384-issue-2311-primary` の `ade323f36` をpushしました。[PR #2400](https://github.com/Motoki0705/tennis-lab/pull/2400)

- 証拠Issueを個別GETで照合し、ラベル付き一覧への反映前でも登録できるようにしました。再試行時も保存済みのIssue番号を使い、重複を検出します。
- `register` と `handoff --accept` は、担当保存後に証拠処理が失敗しても掲示板名とHerdr表示先を同期し、`parent_action_applied: true` と復旧手順を返します。欠落した記録は同じ操作の再実行で1件だけ回収できます。
- 改善候補から取り出した親で `stop --child <親番号>` の後に `create-child` が通ることと、要件節や未知節を変えると拒否されることを確認しました。

関連テスト2,137件は、worker待機の単独再実行とagyの実行環境の分離を含め、すべて通過しました。世代変更テストは実際の `release` と `register` で担当を取り直す形に更新しました。pre-commitのRuffとmypyも通過し、Python CIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2400,"head":"ade323f36e85e6a663c2b3ac04ee9fe7d961861d","resume_on_success":false}}