## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2449,
  "head": "f58984e7b41d4559d976b365a44559403d83e791",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45227001842,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45227001842/job/137876399509",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-23T17:35:07.8970373Z 4.40s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[permission]\n2026-11-23T17:35:07.8971033Z 4.39s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[missing_option]\n2026-11-23T17:35:07.8975132Z FAILED tests/unit/utils/configuration/test_contracts.py::test_all_inventoried_runtime_boundaries_expose_truthful_authorities - AssertionError: assert {'src.submodu...le:main', ...} == {'src.submodu...le:main', ...}\n2026-11-23T17:35:07.9011297Z         'src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples:main',\n2026-11-23T17:35:08.2090861Z ===== 1 failed, 3276 passed, 45 skipped, 253 warnings in 652.13s (0:10:52) =====\n2026-11-23T17:35:09.9697141Z Selected: 3322; outcomes: `{\"failed\": 1, \"passed\": 3276, \"skipped\": 45}`.\n2026-11-23T17:35:09.9718394Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:35:09.9875912Z ##[error]Process completed with exit code 1.\n2026-11-23T17:35:10.6433597Z Selected: 3322; outcomes: `{\"failed\": 1, \"passed\": 3276, \"skipped\": 45}`.\n2026-11-23T17:35:10.6460192Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:35:10.6626103Z ##[error]Process completed with exit code 2.\n2026-11-23T17:35:13.7815662Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (5)",
      "run_id": 45227001842,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45227001842/job/137876399633",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-23T17:40:07.3318101Z tests/unit/utils/configuration/test_audit.py:22: AssertionError\n2026-11-23T17:40:07.3343112Z 13.40s call     tests/e2e/agents_ops/test_parent_completion_records.py::test_recorded_completion_and_failed_proposal_publication_are_independent[parent_self]\n2026-11-23T17:40:07.3377602Z 9.64s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[relation]\n2026-11-23T17:40:07.7566005Z FAILED tests/unit/utils/configuration/test_audit.py::test_repository_source_satisfies_current_audit - AssertionError: assert False\n2026-11-23T17:40:07.7570896Z =========== 1 failed, 2194 passed, 219 warnings in 936.27s (0:15:36) ===========\n2026-11-23T17:40:10.0441809Z Selected: 2195; outcomes: `{\"failed\": 1, \"passed\": 2194}`.\n2026-11-23T17:40:10.0469007Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:40:10.0634868Z ##[error]Process completed with exit code 1.\n2026-11-23T17:40:10.8499073Z Selected: 2195; outcomes: `{\"failed\": 1, \"passed\": 2194}`.\n2026-11-23T17:40:10.8525320Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:40:10.8699505Z ##[error]Process completed with exit code 2.\n2026-11-23T17:40:12.5781036Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45227001842,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45227001842/job/137890983561",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-23T17:46:36.1459379Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-23T17:46:36.1510298Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-23T17:46:36.1556722Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-23T17:46:36.1581652Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-23T17:46:36.1755332Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-23T17:46:38.8637985Z   digest-mismatch: error\n2026-11-23T17:46:40.1903682Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-23T17:46:52.4277560Z     \"incomplete/failed test observations\",\n2026-11-23T17:46:52.4340970Z     \"failed\": 2\n2026-11-23T17:46:52.4539850Z ##[error]Process completed with exit code 1.\n2026-11-23T17:46:52.6555312Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
