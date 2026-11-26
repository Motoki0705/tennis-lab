## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2500,
  "head": "67dcbc42bb7e83f55298152ee7c3d7ac11c0760f",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (6)",
      "run_id": 45618457128,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45618457128/job/139220205222",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "_verified_noop_without_body_edit - shutil.Error: [('[local path]', '[local path]', \"[Errno 2] No such file or directory: '[local path]'\")]\nepair_flow_without_github_writes - shutil.Error: [('[local path]', '[local path]', \"[Errno 2] No such file or directory: '[local path]'\")]\nn_are_not_supplemented[proposal] - shutil.Error: [('[local path]', '[local path]', \"[Errno 2] No such file or directory: '[local path]'\")]\nare_not_supplemented[extraction] - shutil.Error: [('[local path]', '[local path]', \"[Errno 2] No such file or directory: '[local path]'\")]\n2026-11-25T13:08:06.6570923Z ==== 3118 passed, 10 skipped, 264 warnings, 100 errors in 676.03s (0:11:16) ====\n2026-11-25T13:08:09.4013206Z Selected: 3228; outcomes: `{\"failed\": 100, \"passed\": 3118, \"skipped\": 10}`.\n2026-11-25T13:08:09.4036042Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T13:08:09.4212023Z ##[error]Process completed with exit code 1.\n2026-11-25T13:08:10.3401943Z Selected: 3228; outcomes: `{\"failed\": 100, \"passed\": 3118, \"skipped\": 10}`.\n2026-11-25T13:08:10.3425665Z | Fixture | Executions | Setup s | Errors |\n2026-11-25T13:08:10.3641201Z ##[error]Process completed with exit code 2.\n2026-11-25T13:08:14.4369884Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45618457128,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45618457128/job/139233467362",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-25T13:19:22.1873277Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-25T13:19:22.1979739Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-25T13:19:22.2060284Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-25T13:19:22.2103880Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-25T13:19:22.2399130Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-25T13:19:25.4834592Z   digest-mismatch: error\n2026-11-25T13:19:26.4763450Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-25T13:19:39.1274413Z     \"incomplete/failed test observations\",\n2026-11-25T13:19:39.1347776Z     \"failed\": 100\n2026-11-25T13:19:39.1491271Z ##[error]Process completed with exit code 1.\n2026-11-25T13:19:39.3407414Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
