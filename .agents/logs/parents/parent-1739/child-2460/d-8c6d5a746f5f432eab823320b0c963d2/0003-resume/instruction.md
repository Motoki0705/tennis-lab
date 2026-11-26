## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2471,
  "head": "61fef16091952b5c0df06f89b32603656a29a745",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (3)",
      "run_id": 45337862768,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45337862768/job/138254346178",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-24T06:30:34.3074276Z 12.44s call     tests/e2e/agents_ops/test_parent_session.py::test_rechecking_revokes_agreement_and_missing_confirmation_fails\n2026-11-24T06:30:34.3103270Z 5.56s call     tests/e2e/agents_ops/test_issue_supplement.py::test_hook_recovery_records_commands_and_keeps_failures_private[timeout]\n2026-11-24T06:30:34.3115265Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_current_fp32_cache_is_adopted_in_execute_and_load_modes - AssertionError: assert '31f72e2b2cf4...49893394b092c' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-24T06:30:34.3117882Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_batched_tensorrt_fp32_compatibility_requires_the_exact_sources - AssertionError: assert '31f72e2b2cf4...49893394b092c' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-24T06:30:34.3120046Z ========== 2 failed, 6566 passed, 273 warnings in 1403.58s (0:23:23) ===========\n2026-11-24T06:30:36.6629561Z Selected: 6568; outcomes: `{\"failed\": 2, \"passed\": 6566}`.\n2026-11-24T06:30:36.6658419Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T06:30:36.6817232Z ##[error]Process completed with exit code 1.\n2026-11-24T06:30:37.7687565Z Selected: 6568; outcomes: `{\"failed\": 2, \"passed\": 6566}`.\n2026-11-24T06:30:37.7711201Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T06:30:37.7891905Z ##[error]Process completed with exit code 2.\n2026-11-24T06:30:39.5399569Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45337862768,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45337862768/job/138264341739",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-24T06:31:02.3726448Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-24T06:31:02.4025789Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-24T06:31:02.4284281Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-24T06:31:02.4426129Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-24T06:31:02.5339569Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-24T06:31:05.8957280Z   digest-mismatch: error\n2026-11-24T06:31:06.8928351Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-24T06:31:20.7307060Z     \"incomplete/failed test observations\",\n2026-11-24T06:31:20.7388626Z     \"failed\": 2\n2026-11-24T06:31:20.7534021Z ##[error]Process completed with exit code 1.\n2026-11-24T06:31:20.8283786Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
