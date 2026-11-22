PR #2400 のCI失敗を修正し、`bba9f8af5` を `codex/parent-2384-issue-2311-primary` にpushしました。PR本文に原因と検証結果を追記しました。[PR #2400](https://github.com/Motoki0705/tennis-lab/pull/2400)

完了復帰テストでは、dispatcher予約に手動登録の開始経路が残り、世代変更後の `claim` が担当開始記録の欠落で拒否されていました。テストの担当生成を実際のdispatcherと同じ処理に揃え、保存済みの実行根拠と検証結果で復帰できることを確認しました。

失敗したケースを含む関連146件とpre-commitのRuff・mypyが通過しました。新HEADのPython CIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2400,"head":"bba9f8af518f419bfadd9f0e3bf1cb04da56db48","resume_on_success":false}}