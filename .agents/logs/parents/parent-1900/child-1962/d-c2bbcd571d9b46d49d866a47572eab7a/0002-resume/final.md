[PR #1973](https://github.com/Motoki0705/tennis-lab/pull/1973) を作成し、レビュー可能にしました。日本語本文に `Closes #1962` を記載しています。

- branch: `codex/log-turn-traceability`
- HEAD: `0c240f594841272c19c7784007f1d42dce35b9ef`
- 最新HEADの[CI #4085](https://github.com/Motoki0705/tennis-lab/actions/runs/43392046634)：全10 CPU・Lint・system dependency・集約Python成功
- 関連unit/integration：**223 passed（37.95秒）**。pre-commit成功。worktreeはcleanです。

固定監査の **51／4／47、イベントのみ2、和集合53** を再現しました。生成は公開snapshotだけを参照し、公開済みログ原文は変更していません。

| 項目 | 主な検証テスト |
|---|---|
| 1 集計 | `test_fixed_audit_has_separate_published_and_union_denominators` |
| 2 型付き観測・4状態 | `test_four_states_are_counted_once_per_turn`、`test_failed_worker_with_a_final_is_observed_without_publication_success` |
| 3 公開先 | `test_intermediate_prs_and_later_main_turn_stay_distinct` |
| 4 過去事例 | `test_historical_explicit_sources_fix_missing_reports_without_rewriting_events` |
| 5 終了・引継ぎ・再照合 | `test_typed_termination_after_prior_publication_recovers_without_rewriting_history`、`test_handoff_and_unverified_acceptance_are_not_completed_by_runtime_end` |
| 6 英日中README | `test_relative_links_resolve`、`test_decision_record_entry_is_reachable_from_operational_docs` |

全E2Eの初回実装HEADでの結果は **3,926 passed／8 failed／1 skipped（3,439.33秒）**。今回のREADME期待値1件は修正し、publication E2E全42件が成功しました。5件は変更前baseでも同じ失敗を再現。timeout検証2件はbase・変更後の単独実行では成功し、並行時の失敗原因は未確認です。詳細はPRに残しました。

#1961接続用の境界は `trace_capture.capture_termination(operation, receipt, evidence=..., acceptance=...)` です。fixtureでは、worker残件の公開キーは `d-` 付き、decision_log残件は `unpublished_count`[local path]`batch_key`、担当は公開用文字列と仮定しています。契約確定後、このadapterで実際の保存先・型を正規化し、正本の受入・終了根拠を渡す必要があります。

指示どおりrelease／handoff／supervisorへの配線は未実施です。実ログへの限定観測の取り込み・公開と、merge後mainの表示確認も後続作業として残しています。