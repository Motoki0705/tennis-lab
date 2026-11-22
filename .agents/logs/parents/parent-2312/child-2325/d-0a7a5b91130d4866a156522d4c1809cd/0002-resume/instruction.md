## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2340,
  "head": "abddf59c98a734d6874a6c9d325f842987a02540",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 44762835893,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/44762835893/job/136379808463",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-20T15:36:13.7138785Z FAILED tests/e2e/development/test_ci_test_lanes.py::test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs[code-pull_request] - AssertionError: assert {'.agents/log...ADME.md', ...} == {'.agents/log...ADME.md', ...}\n2026-11-20T15:36:13.7143149Z FAILED tests/e2e/development/test_ci_test_lanes.py::test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs[code-push] - AssertionError: assert {'.agents/log...ADME.md', ...} == {'.agents/log...ADME.md', ...}\n2026-11-20T15:36:13.7147973Z FAILED tests/e2e/development/test_ci_test_lanes.py::test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs[mixed-pull_request] - AssertionError: assert {'.agents/log...ADME.md', ...} == {'.agents/log...ADME.md', ...}\n2026-11-20T15:36:14.0780071Z FAILED tests/e2e/development/test_ci_test_lanes.py::test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs[mixed-push] - AssertionError: assert {'.agents/log...ADME.md', ...} == {'.agents/log...ADME.md', ...}\n2026-11-20T15:36:14.0793622Z ========== 16 failed, 1440 passed, 224 warnings in 649.67s (0:10:49) ===========\n2026-11-20T15:36:16.1036976Z Selected: 1456; outcomes: `{\"failed\": 16, \"passed\": 1440}`.\n2026-11-20T15:36:16.1063082Z | Fixture | Executions | Setup s | Errors |\n2026-11-20T15:36:16.1220530Z ##[error]Process completed with exit code 1.\n2026-11-20T15:36:16.7973762Z Selected: 1456; outcomes: `{\"failed\": 16, \"passed\": 1440}`.\n2026-11-20T15:36:16.7999523Z | Fixture | Executions | Setup s | Errors |\n2026-11-20T15:36:16.8158339Z ##[error]Process completed with exit code 2.\n2026-11-20T15:36:20.4975769Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 44762835893,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/44762835893/job/136390445290",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-20T15:42:36.1164467Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-20T15:42:36.1261533Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-20T15:42:36.1341449Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-20T15:42:36.1392579Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-20T15:42:36.1665699Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-20T15:42:39.2249887Z   digest-mismatch: error\n2026-11-20T15:42:40.0268544Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-20T15:42:54.1287084Z     \"incomplete/failed test observations\",\n2026-11-20T15:42:54.1387290Z     \"failed\": 16\n2026-11-20T15:42:54.1552573Z ##[error]Process completed with exit code 1.\n2026-11-20T15:42:54.1758500Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
