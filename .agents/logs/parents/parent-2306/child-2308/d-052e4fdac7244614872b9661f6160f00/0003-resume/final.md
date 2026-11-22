正式レビューのmajor 6件を修正し、`origin/main` にrebaseして既存のドラフトPRへpushしました。PR本文に設計判断と検証結果を追記しました。[PR #2313](https://github.com/Motoki0705/tennis-lab/pull/2313)

**Branch** `codex/parent-2306-issue-2308-primary`  
**HEAD** `3a49c69a6117632e035d60e6b48e6277487ab382`

| 指摘 | 修正と通過した回帰テスト |
| --- | --- |
| 1. 別PRの監視で部分再開が停止 | 事前検査をレビューrun内の全PRに揃え、起動後のリース競合は対象PRと理由を `parent_wait` に保存しました。`test_partial_continue_reports_a_sibling_pr_lease_conflict` |
| 2. 他担当が枠を使うと自動再開を断念 | queue待機中は他担当が枠を使えます。CLIは完了結果を保存し、空き枠を取得してから一度だけ再開します。枠以外の起動拒否は `needs_parent` に理由を残します。`test_auto_resume_retries_after_another_worker_releases_the_only_slot`、`test_auto_resume_records_admission_refusal_for_parent` |
| 3. queueの再待機と完了記録の欠落 | `running/` から `jobs/` への移動を再読で拾い、終端ファイルの観測後60秒を過ぎても完了記録が得られなければエラーを返します。`test_queue_requeue_between_active_directory_reads`、`test_queue_waits_for_bounded_completion_publication`、`test_queue_missing_completion_is_returned_by_wait_worker` |
| 4. PID確認と監視終了の競合 | PID不在時に同じ待機の保存状態を再読し、確定済みの終了結果を返します。`test_controller_exit_rechecks_durable_followup` |
| 5. 長時間監視がmainの更新を妨げる | 監視サービスを固定HEADの一時checkoutで起動し、待機中はmainの保守ロックを解放します。`test_queue_monitor_pins_main_code_without_holding_checkout_lock` |
| 6. 共有memoryに旧手順が残る | 旧手順をmemory・索引・SUMMARYから削除し、Colabの起動時の安定確認とその後のイベント監視を `parent-session` に移しました。`test_local_queue_policy_has_one_operational_source_and_preserves_colab` |

回帰14件を含む関連e2e・unit **575件が通過**しました（413.91秒）。最終HEADでも文書・手動委任の受入と回帰 **232件が通過**しました（96.50秒）。Ruff、mypy、pre-commitも通過しています。実際のsystemd user serviceで、呼び出し元の終了後も監視が続くこと、その間にmainを更新できること、サービス停止時に一時checkoutが回収されることを確認しました。

未確認は、対象HEADのGitHub CIと実学習ジョブを使った一連の自動再開です。