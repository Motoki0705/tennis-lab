## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2399,
  "head": "495767a79a3e642f7aca7b0c99a025558ab52d34",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45025688028,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45025688028/job/137218257089",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T13:14:53.0196131Z 4.06s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[archived]\n2026-11-22T13:14:53.0197441Z 4.05s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[duplicate]\n2026-11-22T13:14:53.3429330Z 4.04s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[missing]\n2026-11-22T13:14:53.3440203Z FAILED tests/e2e/agents_ops/test_worker_log_recovery.py::test_handoff_during_first_worker_merge_stops_later_journals[supervisor_active] - KeyError: 'runs'\n2026-11-22T13:14:53.3440790Z =========== 1 failed, 1582 passed, 201 warnings in 436.33s (0:07:16) ===========\n2026-11-22T13:14:54.8728035Z Selected: 1583; outcomes: `{\"failed\": 1, \"passed\": 1582}`.\n2026-11-22T13:14:54.8745142Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T13:14:54.8883460Z ##[error]Process completed with exit code 1.\n2026-11-22T13:14:55.3975693Z Selected: 1583; outcomes: `{\"failed\": 1, \"passed\": 1582}`.\n2026-11-22T13:14:55.4003569Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T13:14:55.4159645Z ##[error]Process completed with exit code 2.\n2026-11-22T13:14:58.7391840Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45025688028,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45025688028/job/137224877226",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T13:31:36.1285235Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-22T13:31:36.1359481Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-22T13:31:36.1418158Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-22T13:31:36.1449580Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-22T13:31:36.1645603Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-22T13:31:40.6437906Z   digest-mismatch: error\n2026-11-22T13:31:41.3413119Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-22T13:31:51.3319549Z     \"incomplete/failed test observations\",\n2026-11-22T13:31:51.3535870Z     \"failed\": 1\n2026-11-22T13:31:51.3560068Z ##[error]Process completed with exit code 1.\n2026-11-22T13:31:51.3713846Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
