[PR #1971](https://github.com/Motoki0705/tennis-lab/pull/1971) に2件とも修正をpushしました。

branch: `codex/issue-proposal-label`  
修正commit／HEAD: `188e1fca438185b358fd269b5a97ad8ad584cc19`

- **pr-1971-1**: receiptの親番号索引から承認拘束を判定し、本文の出典削除・改変、索引取得失敗を拒否します。共通実行gate、dispatcher、supervisorへ接続しました。
- **pr-1971-2**: 案1を採用し、未掲載候補だけrevisionを固定しました。掲載済み候補は最新revisionへ再投影し、旧承認を無効にするため、本文と承認対象の整合を保てます。

追加テストは次の9ケースです。

- `test_removed_extraction_marker_before_first_agreement_cannot_unbind_approval`
- `test_dispatcher_collection_and_guard_reject_changed_extraction_provenance`（3ケース）
- `test_extraction_index_failure_never_means_an_unbound_parent`（3ケース）
- `test_supervisor_rechecks_numeric_extraction_binding_before_popen`
- `test_partial_proposal_membership_recovers_after_published_candidate_revision`

各修正を個別に戻し、索引照合では5ケース、掲載復旧では1ケースが失敗することを確認して復元しました。索引照合を戻すと、撤回済みparentへの初回 `basis-record` が成功する不具合も再現しました。

関連回帰は **617 passed／729.12秒**。pre-commitのruff・mypyも成功しました。[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43569078247) は **Python・Lint・CPU全10枠・system dependency testsすべて成功**です。証拠はPR本文へ反映済みで、worktreeはcleanです。

再レビュー・merge、および実GitHubでの代表フローとunit稼働確認は親担当として残しています。