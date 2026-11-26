## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2461,
  "head": "a9f3f50f66a2d085a4f6695a0f580d3c39f2b78e",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45326506743,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45326506743/job/138217480094",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-24T04:39:53.2357436Z 4.21s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[missing_option]\n2026-11-24T04:39:53.2358242Z 4.19s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[duplicate]\n2026-11-24T04:39:53.2363284Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_current_fp32_cache_is_adopted_in_execute_and_load_modes - AssertionError: assert 'b6c28c2c19e1...19b659a4eaa8b' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-24T04:39:53.2366755Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_batched_tensorrt_fp32_compatibility_requires_the_exact_sources - AssertionError: assert 'b6c28c2c19e1...19b659a4eaa8b' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-24T04:39:53.2369436Z ===== 2 failed, 3292 passed, 14 skipped, 252 warnings in 648.43s (0:10:48) =====\n2026-11-24T04:39:55.0416057Z Selected: 3308; outcomes: `{\"failed\": 2, \"passed\": 3292, \"skipped\": 14}`.\n2026-11-24T04:39:55.0431725Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T04:39:55.0583200Z ##[error]Process completed with exit code 1.\n2026-11-24T04:39:55.7069697Z Selected: 3308; outcomes: `{\"failed\": 2, \"passed\": 3292, \"skipped\": 14}`.\n2026-11-24T04:39:55.7096904Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T04:39:55.7263575Z ##[error]Process completed with exit code 2.\n2026-11-24T04:39:57.4035634Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45326506743,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45326506743/job/138225913163",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-24T04:50:58.9254168Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-24T04:50:58.9349637Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-24T04:50:58.9439931Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-24T04:50:58.9485609Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-24T04:50:58.9801357Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-24T04:51:02.2158046Z   digest-mismatch: error\n2026-11-24T04:51:02.9380799Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-24T04:51:16.9009117Z     \"incomplete/failed test observations\",\n2026-11-24T04:51:16.9107259Z     \"failed\": 2\n2026-11-24T04:51:16.9264417Z ##[error]Process completed with exit code 1.\n2026-11-24T04:51:16.9467324Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
