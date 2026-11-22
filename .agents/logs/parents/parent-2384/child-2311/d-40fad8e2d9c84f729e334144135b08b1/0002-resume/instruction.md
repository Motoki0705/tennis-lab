## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2400,
  "head": "ade323f36e85e6a663c2b3ac04ee9fe7d961861d",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (0)",
      "run_id": 45025698685,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45025698685/job/137218310256",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T13:16:42.0864225Z 5.69s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[after-create]\n2026-11-22T13:16:42.0870183Z 3.43s call     tests/e2e/agents_ops/test_wait_worker.py::test_wait_uses_latest_run_and_releases_locks_even_with_matching_command_line[resume-7-runtime_error]\n2026-11-22T13:16:42.0871461Z 3.25s call     tests/e2e/agents_ops/test_board_doorbell.py::test_held_failure_after_register_keeps_assignment_and_reports_next_start\n2026-11-22T13:16:42.3810701Z FAILED tests/e2e/agents_ops/test_adopted_proposal_issues.py::test_task_provenance_requires_separate_autonomous_basis_and_current_adoption - AssertionError: Regex pattern did not match.\n2026-11-22T13:16:42.3812302Z ===== 1 failed, 1478 passed, 4 skipped, 247 warnings in 505.08s (0:08:25) ======\n2026-11-22T13:16:43.8559922Z Selected: 1483; outcomes: `{\"failed\": 1, \"passed\": 1478, \"skipped\": 4}`.\n2026-11-22T13:16:43.8581883Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T13:16:43.8734360Z ##[error]Process completed with exit code 1.\n2026-11-22T13:16:44.3740270Z Selected: 1483; outcomes: `{\"failed\": 1, \"passed\": 1478, \"skipped\": 4}`.\n2026-11-22T13:16:44.3769456Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T13:16:44.3924322Z ##[error]Process completed with exit code 2.\n2026-11-22T13:16:47.5226503Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45025698685,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45025698685/job/137224447317",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T13:30:01.5752296Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-22T13:30:01.5817368Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-22T13:30:01.5851439Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-22T13:30:01.6055811Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-22T13:30:04.2835687Z   digest-mismatch: error\n2026-11-22T13:30:05.2722021Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-22T13:30:17.3290809Z     \"incomplete/failed test observations\",\n2026-11-22T13:30:17.3490994Z         \"failed\": 1\n2026-11-22T13:30:17.3533570Z     \"failed\": 1,\n2026-11-22T13:30:17.3559927Z ##[error]Process completed with exit code 1.\n2026-11-22T13:30:17.5406354Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
