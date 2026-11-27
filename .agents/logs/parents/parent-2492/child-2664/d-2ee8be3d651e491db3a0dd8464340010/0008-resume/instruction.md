## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2681,
  "head": "46c25d1b499f3f6367bcae8b6fbd758643ce3b3f",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (4)",
      "run_id": 45918392208,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45918392208/job/140243165764",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-27T00:55:02.0275117Z 6.86s call     tests/e2e/agents_ops/test_progression_cli.py::test_failed_or_incomplete_reviewer_does_not_merge[runtime_failure]\n2026-11-27T00:55:02.0276007Z 6.52s call     tests/e2e/agents_ops/test_parent_session.py::test_child_key_conflict_and_duplicate_existing_fail_closed\n2026-11-27T00:55:02.2536760Z 4.91s call     tests/e2e/agents_ops/test_project_progress.py::test_schema_identity_and_permissions_fail_before_writes[missing]\n2026-11-27T00:55:02.2539778Z FAILED tests/unit/submodules/models/test_public_api.py::test_models_root_owns_the_documented_public_symbols - AssertionError: assert {'BaseInferen...tractor', ...} == {'BaseInferen...tractor', ...}\n2026-11-27T00:55:02.2551436Z ===== 1 failed, 1761 passed, 1 skipped, 234 warnings in 593.32s (0:09:53) ======\n2026-11-27T00:55:03.7400368Z Selected: 1763; outcomes: `{\"failed\": 1, \"passed\": 1761, \"skipped\": 1}`.\n2026-11-27T00:55:03.7419026Z | Fixture | Executions | Setup s | Errors |\n2026-11-27T00:55:03.7562446Z ##[error]Process completed with exit code 1.\n2026-11-27T00:55:04.4107768Z Selected: 1763; outcomes: `{\"failed\": 1, \"passed\": 1761, \"skipped\": 1}`.\n2026-11-27T00:55:04.4125802Z | Fixture | Executions | Setup s | Errors |\n2026-11-27T00:55:04.4287999Z ##[error]Process completed with exit code 2.\n2026-11-27T00:55:08.2193410Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45918392208,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45918392208/job/140251420629",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-27T01:10:19.3318136Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-27T01:10:19.3570125Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-27T01:10:19.3785166Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-27T01:10:19.3899938Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-27T01:10:19.4775274Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-27T01:10:22.5312821Z   digest-mismatch: error\n2026-11-27T01:10:23.4704013Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-27T01:10:36.8360872Z     \"incomplete/failed test observations\",\n2026-11-27T01:10:36.8428093Z     \"failed\": 1\n2026-11-27T01:10:36.8606465Z ##[error]Process completed with exit code 1.\n2026-11-27T01:10:36.8776713Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
