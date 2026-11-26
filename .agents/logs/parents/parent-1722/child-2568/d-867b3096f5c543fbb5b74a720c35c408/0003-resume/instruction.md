## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2569,
  "head": "aac549c1181e8e34105a00d54fabe8f93f8ed700",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45670597575,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45670597575/job/139404874750",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-25T18:14:12.5462632Z 8.89s call     tests/e2e/agents_ops/test_progression_cli.py::test_failed_or_incomplete_reviewer_does_not_merge[output]\n2026-11-25T18:14:12.5465712Z 8.07s call     tests/e2e/agents_ops/test_parent_session.py::test_child_key_conflict_and_duplicate_existing_fail_closed\n2026-11-25T18:14:12.5486331Z FAILED tests/unit/utils/configuration/test_contracts.py::test_all_inventoried_runtime_boundaries_expose_truthful_authorities - AssertionError: assert {'src.submodu...le:main', ...} == {'src.submodu...le:main', ...}\n2026-11-25T18:14:12.5533426Z         'src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples:main',\n2026-11-25T18:14:12.9351043Z ===== 1 failed, 1884 passed, 35 skipped, 226 warnings in 745.05s (0:12:25) =====\n2026-11-25T18:14:15.0469125Z Selected: 1920; outcomes: `{\"failed\": 1, \"passed\": 1884, \"skipped\": 35}`.\n2026-11-25T18:14:15.0495857Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T18:14:15.0660459Z ##[error]Process completed with exit code 1.\n2026-11-25T18:14:15.8383379Z Selected: 1920; outcomes: `{\"failed\": 1, \"passed\": 1884, \"skipped\": 35}`.\n2026-11-25T18:14:15.8410537Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T18:14:15.8593083Z ##[error]Process completed with exit code 2.\n2026-11-25T18:14:17.5590681Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (5)",
      "run_id": 45670597575,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45670597575/job/139404874802",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-25T18:16:09.1278418Z 10.34s call     tests/e2e/tennis_scene/test_video_court_cli.py::test_single_clip_cli_loads_pinned_shared_geometry_or_records_failure[False]\n2026-11-25T18:16:09.1284015Z 9.39s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[relation]\n2026-11-25T18:16:09.1291401Z 8.43s call     tests/e2e/tennis_scene/test_video_court_cli.py::test_single_clip_cli_loads_pinned_shared_geometry_or_records_failure[True]\n2026-11-25T18:16:09.5936604Z FAILED tests/unit/utils/configuration/test_audit.py::test_repository_source_satisfies_current_audit - AssertionError: assert False\n2026-11-25T18:16:09.5941374Z ===== 1 failed, 2062 passed, 1 skipped, 223 warnings in 854.62s (0:14:14) ======\n2026-11-25T18:16:11.9152387Z Selected: 2064; outcomes: `{\"failed\": 1, \"passed\": 2062, \"skipped\": 1}`.\n2026-11-25T18:16:11.9179227Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T18:16:11.9346090Z ##[error]Process completed with exit code 1.\n2026-11-25T18:16:12.7033573Z Selected: 2064; outcomes: `{\"failed\": 1, \"passed\": 2062, \"skipped\": 1}`.\n2026-11-25T18:16:12.7061933Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T18:16:12.7271789Z ##[error]Process completed with exit code 2.\n2026-11-25T18:16:14.4385557Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "CPU tests (3)",
      "run_id": 45670597575,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45670597575/job/139404874843",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-25T18:18:18.6113343Z 14.66s call     tests/e2e/agents_ops/test_board.py::test_parent_lifecycle_names_and_board_failure_recovery\n2026-11-25T18:18:18.6119298Z 10.38s call     tests/e2e/agents_ops/test_parent_session.py::test_rechecking_revokes_agreement_and_missing_confirmation_fails\n2026-11-25T18:18:18.6135650Z 5.48s call     tests/e2e/agents_ops/test_issue_supplement.py::test_hook_recovery_records_commands_and_keeps_failures_private[timeout]\n2026-11-25T18:18:18.8176878Z FAILED tests/e2e/development/test_architecture_boundaries.py::test_static_src_import_modules_exist - AssertionError: unresolved repository imports:\n2026-11-25T18:18:18.8179705Z ===== 1 failed, 3679 passed, 10 skipped, 303 warnings in 987.04s (0:16:27) =====\n2026-11-25T18:18:20.4863474Z Selected: 3690; outcomes: `{\"failed\": 1, \"passed\": 3679, \"skipped\": 10}`.\n2026-11-25T18:18:20.4887648Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T18:18:20.5041615Z ##[error]Process completed with exit code 1.\n2026-11-25T18:18:21.2738645Z Selected: 3690; outcomes: `{\"failed\": 1, \"passed\": 3679, \"skipped\": 10}`.\n2026-11-25T18:18:21.2772635Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T18:18:21.2934541Z ##[error]Process completed with exit code 2.\n2026-11-25T18:18:22.8093243Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45670597575,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45670597575/job/139417300832",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-25T18:21:13.8696029Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-25T18:21:13.8774709Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-25T18:21:13.8817468Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-25T18:21:13.9156939Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-25T18:21:17.3204550Z   digest-mismatch: error\n2026-11-25T18:21:18.4200238Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-25T18:21:33.8026623Z     \"incomplete/failed test observations\",\n2026-11-25T18:21:33.8063984Z         \"failed\": 1\n2026-11-25T18:21:33.8115039Z     \"failed\": 3\n2026-11-25T18:21:33.8299836Z ##[error]Process completed with exit code 1.\n2026-11-25T18:21:33.8526111Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
