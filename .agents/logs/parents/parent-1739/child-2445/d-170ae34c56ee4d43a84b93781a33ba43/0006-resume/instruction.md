## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2469,
  "head": "2feea9f8f55e339179450f8fb449315ac0ace8fb",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (6)",
      "run_id": 45796908528,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45796908528/job/139830122377",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-26T10:08:41.1624115Z 7.95s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[progress]\n2026-11-26T10:08:41.1632487Z 6.08s call     tests/unit/tennis_scene/generate_dataset/test_entrypoint.py::test_real_hydra_cli_fails_when_a_clip_fails\n2026-11-26T10:08:41.6360255Z 4.49s call     tests/e2e/agents_ops/test_parent_judgement_workflow.py::test_failed_prelaunch_can_be_retried_without_a_phantom_run[True]\n2026-11-26T10:08:41.6370274Z FAILED tests/integration/tennis_scene/test_fasthmr_recipe.py::test_only_the_exact_standard_recipe_patch_preserves_previous_source_identity - AssertionError: assert {'submodules/...e3c82d2', ...} == {'submodules/...e3c82d2', ...}\n2026-11-26T10:08:41.6384706Z ===== 1 failed, 1599 passed, 8 skipped, 183 warnings in 521.71s (0:08:41) ======\n2026-11-26T10:08:43.8270263Z Selected: 1608; outcomes: `{\"failed\": 1, \"passed\": 1599, \"skipped\": 8}`.\n2026-11-26T10:08:43.8294134Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T10:08:43.8458716Z ##[error]Process completed with exit code 1.\n2026-11-26T10:08:44.4520082Z Selected: 1608; outcomes: `{\"failed\": 1, \"passed\": 1599, \"skipped\": 8}`.\n2026-11-26T10:08:44.4545240Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T10:08:44.4712612Z ##[error]Process completed with exit code 2.\n2026-11-26T10:08:46.3243167Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45796908528,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45796908528/job/139845467392",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-26T10:27:28.7111745Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-26T10:27:28.7166125Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-26T10:27:28.7223572Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-26T10:27:28.7256889Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-26T10:27:28.7477999Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-26T10:27:31.4694608Z   digest-mismatch: error\n2026-11-26T10:27:32.3626238Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-26T10:27:43.7019884Z     \"incomplete/failed test observations\",\n2026-11-26T10:27:43.7095450Z     \"failed\": 1\n2026-11-26T10:27:43.7313178Z ##[error]Process completed with exit code 1.\n2026-11-26T10:27:43.8781098Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
