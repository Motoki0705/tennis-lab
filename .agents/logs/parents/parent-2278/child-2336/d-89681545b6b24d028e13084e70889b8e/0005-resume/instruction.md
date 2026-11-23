## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2373,
  "head": "c07f0ac7a5805a10e034b0fff406869c10b3f57d",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (5)",
      "run_id": 45003726552,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45003726552/job/137150787410",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T08:59:19.9343708Z 8.42s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[relation]\nt_manual_worker_runs.py::test_from_commit_prepares_and_starts_separate_manual_workers - AssertionError: {\"outcome\": \"failure\", \"error\": \"worktree_outputs_on_mirror requires a symlink to [local path] [local path]"}\n2026-11-22T08:59:19.9367651Z   parent-session: storage preparation failed; fix the reported cause and retry the same --from[local path]  +  where 2 = CompletedProcess(args=['[local path]', '-c', '[local path]').returncode\n2026-11-22T08:59:19.9374648Z ===== 1 failed, 3433 passed, 1 skipped, 255 warnings in 1160.60s (0:19:20) =====\n2026-11-22T08:59:22.3861919Z Selected: 3435; outcomes: `{\"failed\": 1, \"passed\": 3433, \"skipped\": 1}`.\n2026-11-22T08:59:22.3896518Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T08:59:22.4072018Z ##[error]Process completed with exit code 1.\n2026-11-22T08:59:23.3286187Z Selected: 3435; outcomes: `{\"failed\": 1, \"passed\": 3433, \"skipped\": 1}`.\n2026-11-22T08:59:23.3312060Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T08:59:23.3500009Z ##[error]Process completed with exit code 2.\n2026-11-22T08:59:24.9614125Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45003726552,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45003726552/job/137155615569",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T08:59:45.6420334Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-22T08:59:45.6465937Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-22T08:59:45.6514775Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-22T08:59:45.6536896Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-22T08:59:45.6697034Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-22T08:59:48.0866350Z   digest-mismatch: error\n2026-11-22T08:59:49.2454716Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-22T09:00:00.2450443Z     \"incomplete/failed test observations\",\n2026-11-22T09:00:00.2537150Z     \"failed\": 1\n2026-11-22T09:00:00.2709042Z ##[error]Process completed with exit code 1.\n2026-11-22T09:00:00.3056666Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
