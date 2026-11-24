正式レビューの4件を修正し、同じブランチ `codex/parent-2474-issue-2427-primary` に push しました。#2477 を含む最新 main を取り込み、親 CLI の README には双方の記述を残しました。 [PR #2481](https://github.com/Motoki0705/tennis-lab/pull/2481)

- HEAD 更新時にレビュー進行を `parent_wait` として保存し、親へ通知して75で終了します。同じ run から再開し、マージを繰り返さずに CI 確認を続けられることを検証しました。
- メインからの監視起動は claim 前に75で拒否します。同じ待機が専用の固定 worktree で完了することを確認しました。
- 手動修正の同期待機と停止処理に30分20秒、メイン由来の `explain_html serve` に60分15秒の上限を設け、期限超過時の lock 解放を確認しました。
- git-sync は送信前の中断を次回に再試行し、送信結果が不明な場合は重複送信を抑えます。

対象テスト156件と pre-commit が通過しました。実機 journal の確認はマージ後の親側確認として PR に手順を記載しています。Windows 通知はモックで検証しました。

完全 HEAD は `fa3b8cb34157d6296386d5bbfc9b8c164196cc30` です。この HEAD の CI 起動を確認しました。 [CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45358023081)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2481,"head":"fa3b8cb34157d6296386d5bbfc9b8c164196cc30","resume_on_success":false}}