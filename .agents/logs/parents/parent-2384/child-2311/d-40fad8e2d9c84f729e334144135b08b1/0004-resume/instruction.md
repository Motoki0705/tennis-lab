## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2400,
  "head": "accffc3d9d342d50d8770d357a8554f0e2e055c5",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (8)",
      "run_id": 45034042719,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45034042719/job/137243964758",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T14:54:25.4462229Z 9.26s setup    tests/e2e/agents_ops/test_parent_completion_resume.py::test_closed_standalone_parent_resumes_same_session_and_finishes[False-failed]\n2026-11-22T14:54:25.8209113Z 8.06s call     tests/e2e/tennis_scene/test_video_court_cli.py::test_single_clip_cli_loads_pinned_shared_geometry_or_records_failure[True]\n2026-11-22T14:54:25.8222506Z 6.63s call     tests/e2e/agents_ops/test_git_sync.py::test_merge_failure_resumes_only_a_consistent_checkout[True]\n2026-11-22T14:54:25.8238943Z FAILED tests/e2e/agents_ops/test_parent_completion_resume.py::test_closed_parent_resume_checks_execution_evidence[generation] - AssertionError: assert None is not None\n2026-11-22T14:54:25.8242369Z ===== 1 failed, 2083 passed, 74 skipped, 165 warnings in 752.75s (0:12:32) =====\n2026-11-22T14:54:27.9495348Z Selected: 2158; outcomes: `{\"failed\": 1, \"passed\": 2083, \"skipped\": 74}`.\n2026-11-22T14:54:27.9521652Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T14:54:27.9676955Z ##[error]Process completed with exit code 1.\n2026-11-22T14:54:28.7477816Z Selected: 2158; outcomes: `{\"failed\": 1, \"passed\": 2083, \"skipped\": 74}`.\n2026-11-22T14:54:28.7501432Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T14:54:28.7684069Z ##[error]Process completed with exit code 2.\n2026-11-22T14:54:30.1999047Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45034042719,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45034042719/job/137250217944",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T15:03:32.3290421Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-22T15:03:32.3336180Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-22T15:03:32.3360088Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-22T15:03:32.3530289Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-22T15:03:34.9231594Z   digest-mismatch: error\n2026-11-22T15:03:35.8915578Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-22T15:03:46.8822653Z     \"incomplete/failed test observations\",\n2026-11-22T15:03:46.8891313Z         \"failed\": 1\n2026-11-22T15:03:46.8922032Z     \"failed\": 1\n2026-11-22T15:03:46.9074934Z ##[error]Process completed with exit code 1.\n2026-11-22T15:03:47.0368401Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
