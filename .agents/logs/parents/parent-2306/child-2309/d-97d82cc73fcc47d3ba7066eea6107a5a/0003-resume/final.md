正式レビューのmajor 5件を修正し、既存ブランチへpushしました。PR本文に回帰テストの結果とマージ後の担当作業を追記しました。 [PR #2316](https://github.com/Motoki0705/tennis-lab/pull/2316)

**HEAD** `81e5263d5cdcff7b48ef58025e84fcd33c96b99d`

1. **補足ランナーとの衝突** — 共有hookを手動親専用の一時ディレクトリへ移しました。`test_hook_recovery_records_commands_and_keeps_failures_private` で、設定を含む固定コミットから禁止コマンドの拒否、監査記録、補足の適用を確認しました。
2. **起動範囲とPython欠落** — 専用ランチャーが `--add-dir` で起動中だけhookを読み込ませます。`test_agy_launcher_scopes_native_hook_and_reports_missing_venv` で非対象の起動と、起動時・hook実行時の `.venv` 欠落を検証しました。
3. **照合失敗の診断** — 非対象と検証エラーを別の例外型に分けました。`test_agy_hook_reports_verification_errors_without_an_empty_reply` と `test_agy_identity_rejects_unverified_callers_before_github` で、hookと `parent status` が照合異常をエラーとして返すことを確認しました。
4. **Herdrの表示先** — 通常同期と登録直後も呼出元のPID・開始時刻を照合します。`test_agy_register_and_sync_refuse_a_different_pane_with_the_same_conversation` で、別HOMEのTUIが同じ会話をロックしていても、誤ったpaneの表示と所有者記録に書き込まないことを確認しました。
5. **共有DBの切替** — `test_shared_board_v5_web_requires_parent_register_before_v6_cutover` で、新Webのv5拒否、既存担当の登録によるv6移行、既存投稿の表示、旧版のv6拒否を確認しました。 [切替手順](https://github.com/Motoki0705/tennis-lab/blob/81e5263d5cdcff7b48ef58025e84fcd33c96b99d/.agents/ops/board/README.md#共有dbをv6へ切り替える)

agy 1.5.13で `--settings` へのhook指定が拒否され、`--add-dir` では同じ会話を再開できることを実測しました。隔離環境では、専用起動で未読件数の提示とHerdrの表示同期が動き、`.venv` のないcheckoutでのprint・ACPはhook起動0回で成功しました。 [Antigravityのhook探索仕様](https://antigravity.google/docs/customization-discovery?tab=cli)

回帰テスト **23件成功**。関連E2Eは **1335件成功、1件skip（任意のPlaywright）、470.33秒**。pre-commitのruff・mypyも通過しました。

CI結果は親が確認します。本番のDB・Web切替、GitHubラベル同期、有償swarmは未確認です。