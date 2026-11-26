## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2462,
  "head": "7cb45ada9de0c42bed8067b6ed96668e043e5e5e",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (5)",
      "run_id": 45326601456,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45326601456/job/138217798171",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-24T04:44:40.7530279Z tests/unit/utils/configuration/test_audit.py:22: AssertionError\n2026-11-24T04:44:40.7556345Z 12.60s call     tests/e2e/agents_ops/test_parent_completion_records.py::test_recorded_completion_and_failed_proposal_publication_are_independent[parent_self]\n2026-11-24T04:44:40.7587022Z 9.06s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[relation]\n2026-11-24T04:44:41.1053392Z FAILED tests/unit/utils/configuration/test_audit.py::test_repository_source_satisfies_current_audit - AssertionError: assert False\n2026-11-24T04:44:41.1056668Z ===== 1 failed, 2170 passed, 2 skipped, 219 warnings in 893.64s (0:14:53) ======\n2026-11-24T04:44:43.1814913Z Selected: 2173; outcomes: `{\"failed\": 1, \"passed\": 2170, \"skipped\": 2}`.\n2026-11-24T04:44:43.1849968Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T04:44:43.2011765Z ##[error]Process completed with exit code 1.\n2026-11-24T04:44:43.9499716Z Selected: 2173; outcomes: `{\"failed\": 1, \"passed\": 2170, \"skipped\": 2}`.\n2026-11-24T04:44:43.9524027Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T04:44:43.9692358Z ##[error]Process completed with exit code 2.\n2026-11-24T04:44:45.7220447Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (4)",
      "run_id": 45326601456,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45326601456/job/138217798210",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-24T04:40:29.8028732Z 4.18s call     tests/integration/tennis_scene/test_video_batch.py::test_interruption_and_device_failure_stop_without_partial_markers_then_resume[failure1-True]\n2026-11-24T04:40:29.8029615Z 4.17s call     tests/integration/tennis_scene/test_video_batch.py::test_interruption_and_device_failure_stop_without_partial_markers_then_resume[failure2-False]\n2026-11-24T04:40:29.8031076Z FAILED tests/unit/utils/configuration/test_contracts.py::test_all_inventoried_runtime_boundaries_expose_truthful_authorities - AssertionError: assert {'src.submodu...le:main', ...} == {'src.submodu...le:main', ...}\n2026-11-24T04:40:29.8067239Z         'src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples:main',\n2026-11-24T04:40:30.0755469Z ===== 1 failed, 3295 passed, 11 skipped, 256 warnings in 645.01s (0:10:45) =====\n2026-11-24T04:40:31.6850341Z Selected: 3307; outcomes: `{\"failed\": 1, \"passed\": 3295, \"skipped\": 11}`.\n2026-11-24T04:40:31.6866101Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T04:40:31.7010504Z ##[error]Process completed with exit code 1.\n2026-11-24T04:40:32.3316820Z Selected: 3307; outcomes: `{\"failed\": 1, \"passed\": 3295, \"skipped\": 11}`.\n2026-11-24T04:40:32.3332915Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T04:40:32.3516495Z ##[error]Process completed with exit code 2.\n2026-11-24T04:40:35.1717514Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45326601456,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45326601456/job/138226562407",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-24T04:52:48.4217939Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-24T04:52:48.4450228Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-24T04:52:48.4654153Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-24T04:52:48.4764828Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-24T04:52:48.5505360Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-24T04:52:51.2620459Z   digest-mismatch: error\n2026-11-24T04:52:52.1286070Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-24T04:53:04.4119251Z     \"incomplete/failed test observations\",\n2026-11-24T04:53:04.4215666Z     \"failed\": 2\n2026-11-24T04:53:04.4382909Z ##[error]Process completed with exit code 1.\n2026-11-24T04:53:04.4665879Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
