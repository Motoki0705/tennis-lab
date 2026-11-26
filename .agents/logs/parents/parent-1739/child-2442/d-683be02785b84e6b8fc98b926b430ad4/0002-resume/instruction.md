## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2448,
  "head": "7c553b73e0b2154b9e8c42074a566063711e7e45",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45223015730,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45223015730/job/137862465280",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-23T17:13:15.5546090Z 3.81s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[wrong_type]\n2026-11-23T17:13:15.5547234Z 3.79s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[missing]\n2026-11-23T17:13:15.5550249Z FAILED tests/unit/utils/configuration/test_contracts.py::test_all_inventoried_runtime_boundaries_expose_truthful_authorities - AssertionError: assert {'src.submodu...le:main', ...} == {'src.submodu...le:main', ...}\n2026-11-23T17:13:15.5576633Z         'src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples:main',\n2026-11-23T17:13:16.0040451Z ===== 1 failed, 3182 passed, 45 skipped, 254 warnings in 617.70s (0:10:17) =====\n2026-11-23T17:13:17.9519057Z Selected: 3228; outcomes: `{\"failed\": 1, \"passed\": 3182, \"skipped\": 45}`.\n2026-11-23T17:13:17.9532820Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:13:17.9710654Z ##[error]Process completed with exit code 1.\n2026-11-23T17:13:18.5560171Z Selected: 3228; outcomes: `{\"failed\": 1, \"passed\": 3182, \"skipped\": 45}`.\n2026-11-23T17:13:18.5585523Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:13:18.5735525Z ##[error]Process completed with exit code 2.\n2026-11-23T17:13:20.2912233Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (5)",
      "run_id": 45223015730,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45223015730/job/137862465370",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-23T17:17:56.4490844Z tests/unit/utils/configuration/test_audit.py:22: AssertionError\n2026-11-23T17:17:56.4520436Z 12.68s call     tests/e2e/agents_ops/test_parent_completion_records.py::test_recorded_completion_and_failed_proposal_publication_are_independent[parent_self]\n2026-11-23T17:17:56.4544149Z 9.43s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[relation]\n2026-11-23T17:17:56.8470485Z FAILED tests/unit/utils/configuration/test_audit.py::test_repository_source_satisfies_current_audit - AssertionError: assert False\n2026-11-23T17:17:56.8474980Z =========== 1 failed, 2348 passed, 211 warnings in 889.74s (0:14:49) ===========\n2026-11-23T17:17:59.0570574Z Selected: 2349; outcomes: `{\"failed\": 1, \"passed\": 2348}`.\n2026-11-23T17:17:59.0597094Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:17:59.0757287Z ##[error]Process completed with exit code 1.\n2026-11-23T17:17:59.8474359Z Selected: 2349; outcomes: `{\"failed\": 1, \"passed\": 2348}`.\n2026-11-23T17:17:59.8511331Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:17:59.8703810Z ##[error]Process completed with exit code 2.\n2026-11-23T17:18:01.5137544Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45223015730,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45223015730/job/137881561574",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-23T17:31:34.4754027Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-23T17:31:34.5058566Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-23T17:31:34.5328935Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-23T17:31:34.5475219Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-23T17:31:34.6118551Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-23T17:31:37.8510063Z   digest-mismatch: error\n2026-11-23T17:31:38.8346881Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-23T17:31:52.9917254Z     \"incomplete/failed test observations\",\n2026-11-23T17:31:53.0296515Z     \"failed\": 2\n2026-11-23T17:31:53.0329271Z ##[error]Process completed with exit code 1.\n2026-11-23T17:31:53.1701014Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
