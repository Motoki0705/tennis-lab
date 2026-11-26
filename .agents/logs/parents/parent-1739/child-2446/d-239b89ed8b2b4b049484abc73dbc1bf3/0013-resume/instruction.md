## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2470,
  "head": "f5566a4be572a8dc80a3421ddd306d2a6f8f82e9",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (3)",
      "run_id": 45849564964,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45849564964/job/140012380965",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-26T15:41:10.1592236Z 10.83s call     tests/e2e/agents_ops/test_parent_session.py::test_rechecking_revokes_agreement_and_missing_confirmation_fails\n2026-11-26T15:41:10.1611039Z 5.48s call     tests/e2e/agents_ops/test_issue_supplement.py::test_hook_recovery_records_commands_and_keeps_failures_private[timeout]\n2026-11-26T15:41:10.3509380Z FAILED tests/integration/agents_ops/test_storage_tier.py::test_blocked_writer_is_released_and_lease_break_is_audited[after-final-check] - AssertionError: Traceback (most recent call last):\n2026-11-26T15:41:10.3511771Z   FileNotFoundError: [Errno 2] No such file or directory: '[local path]'\n2026-11-26T15:41:10.3514533Z ===== 1 failed, 2233 passed, 14 skipped, 243 warnings in 618.35s (0:10:18) =====\n2026-11-26T15:41:11.8492249Z Selected: 2248; outcomes: `{\"failed\": 1, \"passed\": 2233, \"skipped\": 14}`.\n2026-11-26T15:41:11.8516147Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T15:41:11.8664652Z ##[error]Process completed with exit code 1.\n2026-11-26T15:41:12.6205374Z Selected: 2248; outcomes: `{\"failed\": 1, \"passed\": 2233, \"skipped\": 14}`.\n2026-11-26T15:41:12.6221873Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T15:41:12.6385504Z ##[error]Process completed with exit code 2.\n2026-11-26T15:41:14.2467122Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45849564964,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45849564964/job/140028253569",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-26T15:58:12.5104847Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-26T15:58:12.5202829Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-26T15:58:12.5287763Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-26T15:58:12.5332727Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-26T15:58:12.5693077Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-26T15:58:16.1188570Z   digest-mismatch: error\n2026-11-26T15:58:17.2787480Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-26T15:58:32.1581248Z     \"incomplete/failed test observations\",\n2026-11-26T15:58:32.1718351Z     \"failed\": 1\n2026-11-26T15:58:32.1882889Z ##[error]Process completed with exit code 1.\n2026-11-26T15:58:32.2091975Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
