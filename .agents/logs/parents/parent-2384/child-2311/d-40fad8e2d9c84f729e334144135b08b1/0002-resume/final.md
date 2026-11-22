CIで失敗した採用提案テストを修正しました。`ParentSession.register()` で担当開始を記録してから、実行根拠が欠けている場合の拒否と、採用状態の判定を検証します。

関連テスト123件とpre-commitのRuff・mypyが通過しました。`codex/parent-2384-issue-2311-primary` に `56c8e3426` をpushし、原因と検証結果を追記しました。新HEADのPython CIは結果待ちです。[PR #2400](https://github.com/Motoki0705/tennis-lab/pull/2400)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2400,"head":"56c8e3426d79c14a406bd1a5b2a946e297d066bf","resume_on_success":false}}