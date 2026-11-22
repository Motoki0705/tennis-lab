#2309 を実装し、pushしてドラフトPRを作成しました。 [PR #2316](https://github.com/Motoki0705/tennis-lab/pull/2316)

- ブランチ `codex/parent-2306-issue-2309-primary`
- HEAD `b86c2436a1509a9252286749a546450cca887d2e`

受入条件は次のとおり確認しました。

1. **ルート識別** — 会話ID・前景TUI・排他ロック・読み取りRPCを照合します。E2Eでprint・ACP・入れ子・祖先を確認できない呼び出しを拒否し、実機で別TUIに開き直したサブエージェントとworkflowの子会話も拒否しました。
2. **親担当** — `register`・`status`・`handoff`・`release` をCLIから実行し、registryの担当と世代、引継ぎ後の前任の操作拒否を確認しました。
3. **Herdr** — agy 1.5.13とHerdr 0.9.5で `[local path]` 時の表示解除、`[local path]` 時の復元、`[local path]` 後の解除を確認しました。重複paneの拒否と、旧paneが空いた後の移動はE2Eで確認しました。
4. **掲示板** — 名簿・在席・投稿・返信・既読を確認しました。`PreInvocation` が担当会話へ未読件数だけを渡し、`awaiting_hook` が配送失敗と区別されることを検証しました。既存の配送履歴を保持するDB移行も通過しました。
5. **追跡規約** — `agent:agy` と `agy/` を追加し、相反する著者ラベルの拒否をテストしました。agy親の対応範囲は手動の対話起動と文書化しました。

ローカルの対象E2Eは **1200件成功、1件skip、330.39秒**。増分は22ケースです。skipは任意のPlaywrightテストで、pre-commitのruffとmypyは成功しました。補足の運用コード型検査に出た8件は `origin/main` でも再現しました。

CI結果、有償swarm、本番での初回起動は未確認です。CIの監視は親へ引き継ぎます。マージ後は共有掲示板DBの移行とGitHubラベルの同期が残ります。