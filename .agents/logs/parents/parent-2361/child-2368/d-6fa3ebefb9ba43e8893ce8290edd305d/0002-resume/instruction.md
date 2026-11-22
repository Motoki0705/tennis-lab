## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2371,
  "head": "99d882a86d7d6ae0499c793c65d974c39509907a",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (3)",
      "run_id": 45000182493,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45000182493/job/137139984606",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T08:04:11.1255176Z 6.97s call     tests/unit/tennis_scene/generate_dataset/test_entrypoint.py::test_real_hydra_cli_fails_when_a_clip_fails\nt_manual_worker_runs.py::test_from_commit_prepares_and_starts_separate_manual_workers - AssertionError: {\"outcome\": \"failure\", \"error\": \"worktree_outputs_on_mirror requires a symlink to [local path] [local path]"}\n2026-11-22T08:04:11.4987074Z   parent-session: storage preparation failed; fix the reported cause and retry the same --from[local path]  +  where 2 = CompletedProcess(args=['[local path]', '-c', '[local path]').returncode\n2026-11-22T08:04:11.4991237Z ===== 1 failed, 2019 passed, 39 skipped, 387 warnings in 606.34s (0:10:06) =====\n2026-11-22T08:04:13.5565809Z Selected: 2059; outcomes: `{\"failed\": 1, \"passed\": 2019, \"skipped\": 39}`.\n2026-11-22T08:04:13.5597010Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T08:04:13.5759410Z ##[error]Process completed with exit code 1.\n2026-11-22T08:04:14.3385813Z Selected: 2059; outcomes: `{\"failed\": 1, \"passed\": 2019, \"skipped\": 39}`.\n2026-11-22T08:04:14.3415397Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T08:04:14.3597483Z ##[error]Process completed with exit code 2.\n2026-11-22T08:04:16.0212548Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45000182493,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45000182493/job/137146266232",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T08:20:06.4153654Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-22T08:20:06.4244889Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-22T08:20:06.4328145Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-22T08:20:06.4371057Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-22T08:20:06.4678651Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-22T08:20:09.7244521Z   digest-mismatch: error\n2026-11-22T08:20:10.6031928Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-22T08:20:25.7137631Z     \"incomplete/failed test observations\",\n2026-11-22T08:20:25.7228055Z     \"failed\": 1\n2026-11-22T08:20:25.7414366Z ##[error]Process completed with exit code 1.\n2026-11-22T08:20:25.7633858Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
