子Issue #2311 の正式レビュー指摘3件を修正し、`codex/parent-2384-issue-2311-primary` にpushしました。PR本文に各コミットと検証結果を追記し、`Closes #2311` を確認しました。[PR #2400](https://github.com/Motoki0705/tennis-lab/pull/2400)

- `6b33afd67` 証拠Issueの作成がHTTP 403で拒否された後、再登録でIssueと `assigned` を各1件だけ作成できるようにしました。応答不明時は重複POSTを防ぎます。
- `adbbdb063` 担当世代ごとに開始経路を記録し、欠けた `assigned` と `handoff` をそれぞれ正しいコマンドで回収します。旧dispatcherの未完了ジョブや未処理の終了通知がある場合は、照合を済ませてから新世代へ切り替えます。
- `accffc3d9` 構造と詳細ハッシュが一致する停止・再開記録だけを承認照合から外します。`##` を含む入力で親を停止した後も分割でき、同じ節に手書きした自由文や `###` の追加要件は拒否されることを確認しました。

関連テストは重複を除く1,106件が通過しました。追加は5件です。3コミットすべてでpre-commitのRuffとmypyが通過しました。異常系はfake GitHub・Herdrで検証し、新HEADのPython CIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2400,"head":"accffc3d9d342d50d8770d357a8554f0e2e055c5","resume_on_success":false}}