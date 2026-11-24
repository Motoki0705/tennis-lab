## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2448,
  "head": "f23c0e0ecd4d208ae0730032548aa46ca10d6f83",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45230261122,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45230261122/job/137887908744",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-23T17:55:37.7206500Z 5.50s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[archived]\n2026-11-23T17:55:37.7209958Z 5.47s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[wrong_type]\n2026-11-23T17:55:37.7214405Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_current_fp32_cache_is_adopted_in_execute_and_load_modes - AssertionError: assert '472d7c9feebf...2d59ef30cbaf5' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-23T17:55:37.7217569Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_batched_tensorrt_fp32_compatibility_requires_the_exact_sources - AssertionError: assert '472d7c9feebf...2d59ef30cbaf5' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-23T17:55:37.7220336Z ===== 2 failed, 3186 passed, 11 skipped, 249 warnings in 795.38s (0:13:15) =====\n2026-11-23T17:55:39.8521074Z Selected: 3199; outcomes: `{\"failed\": 2, \"passed\": 3186, \"skipped\": 11}`.\n2026-11-23T17:55:39.8563311Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:55:39.8690132Z ##[error]Process completed with exit code 1.\n2026-11-23T17:55:40.6687876Z Selected: 3199; outcomes: `{\"failed\": 2, \"passed\": 3186, \"skipped\": 11}`.\n2026-11-23T17:55:40.6714248Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:55:40.6867245Z ##[error]Process completed with exit code 2.\n2026-11-23T17:55:42.3333312Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (8)",
      "run_id": 45230261122,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45230261122/job/137887908785",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-23T17:55:08.3921542Z     name: 'TimeoutError'\n2026-11-23T17:55:08.3922901Z  +  where 1 = CompletedProcess(args=['node', '[local path]'], retu.../work/tennis-lab/tennis-lab/tests/e2e/agents_ops/board_browser.cjs:373:16) {[local path]  log: [],[local path]  name: 'TimeoutError'[local path]}[local path]").returncode\n2026-11-23T17:55:08.3923809Z ===== 1 failed, 2078 passed, 37 skipped, 155 warnings in 758.92s (0:12:38) =====\n2026-11-23T17:55:10.5908108Z Selected: 2116; outcomes: `{\"failed\": 1, \"passed\": 2078, \"skipped\": 37}`.\n2026-11-23T17:55:10.5930117Z | tests/e2e/tennis_scene/test_publication_failures.py | 2 | 0.004 | 14.894 | 0.001 |\n2026-11-23T17:55:10.5931974Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:55:10.6098802Z ##[error]Process completed with exit code 1.\n2026-11-23T17:55:11.3419949Z Selected: 2116; outcomes: `{\"failed\": 1, \"passed\": 2078, \"skipped\": 37}`.\n2026-11-23T17:55:11.3435193Z | tests/e2e/tennis_scene/test_publication_failures.py | 2 | 0.004 | 14.894 | 0.001 |\n2026-11-23T17:55:11.3436365Z | Fixture | Executions | Setup s | Errors |\n2026-11-23T17:55:11.3610236Z ##[error]Process completed with exit code 2.\n2026-11-23T17:55:13.0769850Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45230261122,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45230261122/job/137903470601",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-23T18:06:36.3651789Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-23T18:06:36.3741304Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-23T18:06:36.3787891Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-23T18:06:36.4102686Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-23T18:06:39.6979531Z   digest-mismatch: error\n2026-11-23T18:06:40.6172210Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-23T18:06:55.7319685Z     \"incomplete/failed test observations\",\n2026-11-23T18:06:55.7367702Z         \"failed\": 1\n2026-11-23T18:06:55.7457921Z     \"failed\": 3\n2026-11-23T18:06:55.7649190Z ##[error]Process completed with exit code 1.\n2026-11-23T18:06:55.8490625Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
