# PR #2184 のレビュー指摘とCI失敗の修正（#2176）

統合レビュー（run parent-2157-0c6622a233194ce483d3bdb22f96a068）で major が5件、PR HEAD f5d63d7f9 のCI（run 43802177948）で88件の失敗が出ました。すべて修正し、同じPR（branch `codex/merge-pipeline-switch`）へpushしてください。

## レビュー指摘（major）

### pr-2184-1: 作成workerの有無にかかわらず対象Issueの停止を尊重する
`.agents/ops/review/progression_backend.py`:202

静的推論です。停止検査が `if worker` 内にあるため、親の直接実装では `stop --child N` が保存した tasks[N].stopped と本文の停止記録を無視します。レビュー後のCI待機中に停止しても、CI成功後にmergeできます。対象Issue全件の停止をworkerとの対応確認から分離してください。削除した停止制御テストを厳密照合の廃止として扱った対応表も、この存続すべき契約を取り落としています。

### pr-2184-2: dispatcher経由でも作成workerの修正ターンを起動できること
`.agents/ops/review/progression_backend.py`:313

静的推論です。ここから start_repair → repairs.reserve → Supervisor._job → dispatcher.collect._collect と進むと、seeds指定時も `repos/.../pulls?state=all` を取得します。しかし渡したProgressionGitHubはそのendpointを必ず拒否します。そのためdispatcher親のCI失敗・gate修正はworker予約前に失敗し、recovery_requiredで停止します。実adapterを通す対象限定の取得経路が必要です。追加テストは手動adapterと通常クライアントのdispatcherを別々に検証しており、この接続を通していません。

### pr-2184-3: 同じworkerでmajorを修正した後、親の明示続行で進めること
`.agents/ops/review/progression.py`:234

静的推論です。文書どおりdispatcherの `continue-worker` でmajorを修正すると、同じsessionでも ParentSession.delegate が新しいdelegation_idを発行します。merge-continueは承認状態だけを変更し、保存したworkerの委任IDと修正登録を更新しないため、次のhuman検査が必ず creating_worker_changed_or_stopped になります。再依頼も未完了runを返すため復帰できません。明示続行時に同一workerの継続来歴・終了を確認して対応を更新する経路が必要です。

### pr-2184-4: 切替前にmerge済みの成果を含む親も完了確認できること
`.agents/ops/review/merge_gate.py`:295

静的推論です。development.verify_completionは全成果PRにこの関数を要求しますが、schema 4の完了記録しか受理しません。旧方式でmerge済みの子を含む親（今回の#2157の#2177・#2178を含む）は必ず拒否されます。既にclosedのPRはreview-requestにも登録できないため、後から確認記録を作る入口がありません。旧レビューを新しいmerge権限へ変換せず、既にmerge済みのPRを対象限定で後確認できる移行経路が必要です。

### pr-2184-5: レビューから原要件・合意記録・依存PRを参照できること
`.agents/ops/review/parent_review.py`:78

静的推論です。contextは親・対象・依存Issueの直接コメントだけを読み、従来parent_commentsが含めていた証拠用Issueの合意・判断記録を渡しません。さらに同じ番号集合をMCPの許可範囲にするため、リンクを辿って補完することもFORBIDDEN_SCOPEで拒否されます。保存済みdependenciesのPR番号も許可範囲から漏れています。証拠用Issueにある受入条件や対象外の依存PRを確認できるよう、保存済みの関係から取得範囲を構成してください。

### 指摘への補足（親の判断）

- pr-2184-1: 親の直接実装（作成workerなし）でも、`stop --child N` の停止と人間制御を全段階で尊重してください。削除した旧テストのうち、厳密照合ではなく停止制御の契約を検証していたものは、新しい経路のテストとして復元してください。旧テスト対応表のうち、同様に「存続すべき契約」を落としていないかを見直してください。
- pr-2184-2: dispatcher親の修正起動は、実際のProgressionGitHub adapterを通すe2eで確かめてください。修正起動のためにrepo全体のPR一覧（`pulls?state=all`）を取りに行くのではなく、対象を番号で取る経路にしてください。
- pr-2184-3: majorを同じworkerで直したあと、親の明示続行で進めることが、手動・dispatcherの両方でできるようにしてください。
- pr-2184-4: この親（#2157）の #2177・#2178 のように、切替前の旧方式でmerge済みのPRを含む親も完了確認できる必要があります。旧レビューを新しいmerge権限へ読み替えるのではなく、既にmerge済みのPRを対象限定で後確認（merge commitの存在・main CI・対象Issueの完了）する移行経路にしてください。
- pr-2184-5: 証拠用Issueの合意・判断記録と、保存済み依存のPRを、レビューの取得範囲（MCPの許可範囲を含む）へ入れてください。repo全体の一覧取得には戻さないでください。

## CI失敗（88件）

1. `tests/e2e/agents_ops/test_adopted_proposal_issues.py` 76件: `AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'`。fake GitHubの更新漏れ。
2. `test_parent_development.py` 8件・`test_parent_self_work.py` 2件・`test_parent_completion_records.py` 1件・`test_improvement_recovery.py` 1件: `AssertionError: parent-session: auto_merge_disabled`。手元は `[local path]` で `AGENT_AUTO_MERGE=1` が入っているため通り、CIでは未設定のため失敗していると推測します。テストが開発者の環境変数に依存しないよう、必要な値をテスト内で明示してください（本番の既定値・判定は変えない）。

失敗したテスト一覧（CIログより）:

```
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_lost_shared_parent_journal_stops_before_agent_or_writes_and_can_be_restored[False-empty] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_h
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_lost_shared_parent_journal_stops_before_agent_or_writes_and_can_be_restored[False-missing] - AttributeError: 'FakeGitHub' object has no attribute 'targeted
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_lost_shared_parent_journal_stops_before_agent_or_writes_and_can_be_restored[True-empty] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_hi
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_lost_shared_parent_journal_stops_before_agent_or_writes_and_can_be_restored[True-missing] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_newly_observed_shared_parent_requires_journal_before_organization - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_shared_parent_metadata_must_match_journal_before_any_publication[bindings] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_shared_parent_metadata_must_match_journal_before_any_publication[node] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
ERROR tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_shared_parent_metadata_must_match_journal_before_any_publication[plan] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_adds_child_to_unowned_existing_parent_and_preserves_old_requirements[False] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_adds_child_to_unowned_existing_parent_and_preserves_old_requirements[True] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_adoption_revocation_after_link_is_recorded_once_and_never_deletes_task - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_all_pages_and_api_failures_are_not_replaced_by_empty_inputs - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_ambiguous_create_is_not_repeated_when_the_issue_cannot_be_found - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_autonomous_parent_notes_preserve_adopted_task_delegation[child] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_autonomous_parent_notes_preserve_adopted_task_delegation[parent] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_cancelled_mapped_proposal_does_not_block_independent_new_issue - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_checklist_history_failure_retains_checkpoint_and_reverifies_on_resume - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_checklist_recovers_supplement_between_get_and_patch_once[False] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_checklist_recovers_supplement_between_get_and_patch_once[True] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_checklist_second_interruption_stops_and_resumes_saved_body_without_duplication - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_completed_reuse_still_preserves_active_pr_and_explicit_hold_boundaries[human] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_completed_reuse_still_preserves_active_pr_and_explicit_hold_boundaries[pr] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_context_only_split_tasks_recheck_all_adoptions - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_context_sources_remain_in_split_inputs_after_index_anchors_move - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_corrupt_or_missing_journal_does_not_recreate_managed_issues - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_created_issue_renamed_and_completed_after_lost_response_is_recovered - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_cross_report_group_is_visible_through_unchanged_dispatcher_collection - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_dry_run_validates_plan_without_writing_issues_or_journal - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_edit_after_create_records_reason_and_resumes_same_plan[adoption] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_edit_after_create_records_reason_and_resumes_same_plan[body] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_edit_while_agent_is_running_prevents_issue_creation[adoption] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_edit_while_agent_is_running_prevents_issue_creation[body] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_edit_while_agent_is_running_prevents_issue_creation[option] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_edit_while_agent_is_running_prevents_issue_creation[parent] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_existing_owner_conflict_requires_reasoned_hold[assignee] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_existing_owner_conflict_requires_reasoned_hold[owner] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_existing_owner_conflict_requires_reasoned_hold[pr] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_fenced_task_metadata_example_is_not_an_adopted_task - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_information_hold_is_specific_and_unchanged_input_does_not_rerun_agent - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[confirmation] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[cycle] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[digest] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[duplicate] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[evidence] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[missing] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_invalid_agent_plan_is_rejected_as_a_whole[unknown] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_legacy_standalone_draft_is_preserved_across_resume[created] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_legacy_standalone_draft_is_preserved_across_resume[saved] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_legacy_standalone_draft_is_preserved_across_resume[unknown] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_multiple_proposals_share_parent_and_keep_distinct_index_anchors - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_organization_keeps_concurrent_delegation_and_rejects_old_plan - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_organizer_accepts_completed_run_with_verified_reconnect_notice - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_parent_checklist_race_preserves_supplement_and_journal_for_retry - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_parentless_adoption_creates_normal_standalone_once - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_public_source_check_works_with_only_the_documented_ops_import_root - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_response_loss_at_every_publication_stage_recovers_without_duplicates[repos/o/r/issues/100] - AttributeError: 'FakeGitHub' object has no attribute 'targete
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_response_loss_at_every_publication_stage_recovers_without_duplicates[repos/o/r/issues/101/sub_issues] - AttributeError: 'FakeGitHub' object has no attribu
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_response_loss_at_every_publication_stage_recovers_without_duplicates[repos/o/r/issues/101] - AttributeError: 'FakeGitHub' object has no attribute 'targete
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_response_loss_at_every_publication_stage_recovers_without_duplicates[repos/o/r/issues] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_hi
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_reuses_existing_work_without_reopening_renaming_or_reparenting[False] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_reuses_existing_work_without_reopening_renaming_or_reparenting[True] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_runtime_failure_has_no_github_side_effects[cli_error] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_runtime_failure_has_no_github_side_effects[empty] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_runtime_failure_has_no_github_side_effects[wrong_model] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_selected_option_and_digest_stay_bound_after_the_human_changes_choice - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_source_change_defers_only_its_subject_closure_in_v2[final_check] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_source_change_defers_only_its_subject_closure_in_v2[preflight] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_source_change_defers_only_its_subject_closure_in_v2[publication] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_task_provenance_requires_separate_autonomous_basis_and_current_adoption - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_task_source_metadata_cannot_silently_fall_back_to_legacy[changed_digest] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_task_source_metadata_cannot_silently_fall_back_to_legacy[duplicate] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_task_source_metadata_cannot_silently_fall_back_to_legacy[unknown_version] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_task_source_metadata_cannot_silently_fall_back_to_legacy[wrong_position] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_triage_publishes_only_until_the_scheduled_judgement_entry_runs - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_unselected_or_incomplete_report_never_launches_agent[complete-False] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_unselected_or_incomplete_report_never_launches_agent[incomplete-True] - AttributeError: 'FakeGitHub' object has no attribute 'targeted_history'
FAILED tests/e2e/agents_ops/test_improvement_recovery.py::test_finish_response_loss_repairs_only_missing_record[parent_self-ended] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_completion_records.py::test_recorded_completion_and_failed_proposal_publication_are_independent[parent_self] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_parent_distinguishes_confirmed_cancellation_from_success[body] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_parent_distinguishes_confirmed_cancellation_from_success[missing] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_parent_distinguishes_confirmed_cancellation_from_success[open_pr] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_parent_distinguishes_confirmed_cancellation_from_success[state_reason] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_parent_distinguishes_confirmed_cancellation_from_success[worker] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_report_requires_gate_real_merge_close_and_parent_verification[lost_response-standalone] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_report_requires_gate_real_merge_close_and_parent_verification[not_closed-standalone] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_development.py::test_completed_report_requires_gate_real_merge_close_and_parent_verification[wrong_head-standalone] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_self_work.py::test_self_completion_waits_for_real_evidence[acceptance-parent_self] - AssertionError: parent-session: auto_merge_disabled
FAILED tests/e2e/agents_ops/test_parent_self_work.py::test_self_completion_waits_for_real_evidence[worker-parent_self] - AssertionError: parent-session: auto_merge_disabled
```

## 検証

- CI相当の環境（`AGENT_AUTO_MERGE` などの個人環境変数を外した状態）で、上の失敗ファイルと agents_ops のe2e全体を実行して確認してください。
- 指摘ごとに、修正を外すと失敗するテストを追加してください。
- push後に、指摘ごとの修正内容・追加テスト・テスト結果・新しいHEADを報告して終了してください。CIの完了は待たなくてよい。
- 修正方針が要件から一意に決まらない点があれば、推測で進めず報告で質問してください。
