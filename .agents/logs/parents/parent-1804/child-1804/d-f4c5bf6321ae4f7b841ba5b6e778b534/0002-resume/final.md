[PR #1902](https://github.com/Motoki0705/tennis-lab/pull/1902) を作成・pushしました。

- branch: `codex/parent-self-work-target`
- HEAD: `e705457f0`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42964884509): **Lint・CPU10分割・Pythonすべて成功**。Pythonの対象HEAD一致も確認済み。
- worktreeはclean。merge・Issue close・担当終了は行っていません。

| 受入条件 | 証拠 |
|---|---|
| 1. 親自身を一件だけ対象化 | `test_labelled_parent_resolves_once_and_sync_preserves_requirements` 成功。階層書込み0回。 |
| 2. 直接実装[local path] | `test_self_result_review_merge_completion_without_creating_children`、closed親の復帰・同一session継続も成功。 |
| 3. 未完了状態の拒否・本文保持 | `test_self_completion_waits_for_real_evidence`、`test_postmerge_worker_check_requires_parent_acceptance` 成功。 |
| 4. 既存管理・安全条件 | 分割・取得失敗・移管・二重起動拒否を確認。排他等33件、triage等31件、入れ子親1件成功。 |
| 5. 文書整合 | `pytest tests/e2e/agents_ops/test_agents_docs.py -n 0 -q`：119件成功。skill validatorも成功。 |
| 6. 来歴の接続 | 自身へのworker journalで `source.parent == source.issue`、同一session継続を確認。 |

新規19件と修正対象2件は、次のコマンドで **21 passed／124.40秒**。全コマンド・実測・初回失敗と修正経緯はPR本文に記載しました。

```bash
.venv/bin/python -m pytest \
  tests/e2e/agents_ops/test_parent_self_work.py \
  tests/e2e/agents_ops/test_parent_self_managed.py \
  tests/e2e/agents_ops/test_parent_completion_resume.py::test_labelled_parent_recovers_completion_with_or_without_a_worker \
  tests/e2e/agents_ops/test_review_inspection.py::test_inspect_keeps_other_pr_facts_on_retrieval_failure_and_detects_race \
  tests/e2e/agents_ops/test_parent_development.py::test_workflow_report_rejects_duplicate_json_and_empty_parent_completion \
  -n 2 -q
```

条件7の比較材料は、分割前提なら親＋子の最低2Issueと子作成・関係登録等が必要なところ、本件は**1Issue・子0件・分割操作0回**です。workerは新規1回／継続1回（中断復旧）、正式レビュー登録は現時点0回です。時間・トークン削減率は未測定です。

親ラベル付きの対応はfake GitHubで検証しました。本件自身は親ラベルなしの既存経路で進めています。残る正式レビュー、merge、main CI、全体受入は親へ引き渡します。