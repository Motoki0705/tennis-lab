## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2612,
  "head": "0497bcfe0ee4c7188ad541a387886e373fd5af08",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (8)",
      "run_id": 45792848770,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45792848770/job/139816219342",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-26T09:43:53.4693441Z 4.82s call     tests/integration/tennis_scene/test_video_batch.py::test_interruption_and_device_failure_stop_without_partial_markers_then_resume[failure0-True]\n2026-11-26T09:43:53.4694607Z FAILED tests/e2e/agents_ops/test_board_web.py::test_browser_human_conversation_workflow - AssertionError: page.waitForFunction: Timeout 30000ms exceeded.\n2026-11-26T09:43:53.4695552Z     name: 'TimeoutError'\n2026-11-26T09:43:53.4696555Z  +  where 1 = CompletedProcess(args=['node', '[local path]'], retu.../work/tennis-lab/tennis-lab/tests/e2e/agents_ops/board_browser.cjs:373:16) {[local path]  log: [],[local path]  name: 'TimeoutError'[local path]}[local path]").returncode\n2026-11-26T09:43:53.4697236Z ===== 1 failed, 2429 passed, 35 skipped, 200 warnings in 620.83s (0:10:20) =====\n2026-11-26T09:43:55.3291744Z Selected: 2465; outcomes: `{\"failed\": 1, \"passed\": 2429, \"skipped\": 35}`.\n2026-11-26T09:43:55.3303684Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T09:43:55.3469959Z ##[error]Process completed with exit code 1.\n2026-11-26T09:43:55.8965415Z Selected: 2465; outcomes: `{\"failed\": 1, \"passed\": 2429, \"skipped\": 35}`.\n2026-11-26T09:43:55.8995657Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T09:43:55.9173362Z ##[error]Process completed with exit code 2.\n2026-11-26T09:43:57.8591662Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45792848770,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45792848770/job/139828423669",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-26T09:56:03.4541965Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-26T09:56:03.4598627Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-26T09:56:03.4628133Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-26T09:56:03.4868669Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-26T09:56:07.6125954Z   digest-mismatch: error\n2026-11-26T09:56:08.8064664Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-26T09:56:23.6137768Z     \"incomplete/failed test observations\",\n2026-11-26T09:56:23.6163609Z         \"failed\": 1\n2026-11-26T09:56:23.6209170Z     \"failed\": 1\n2026-11-26T09:56:23.6405676Z ##[error]Process completed with exit code 1.\n2026-11-26T09:56:23.6768089Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
