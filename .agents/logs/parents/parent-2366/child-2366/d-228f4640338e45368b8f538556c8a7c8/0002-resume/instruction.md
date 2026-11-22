## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2376,
  "head": "5f60e3986cc6f57ad92e53eda8d4299b7bf0caa6",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (5)",
      "run_id": 45005616425,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45005616425/job/137156687843",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-22T09:16:50.9059768Z 5.65s call     tests/e2e/agents_ops/test_parent_session.py::test_child_creation_recovers_partial_failure_without_duplicates[relation]\nt_manual_worker_runs.py::test_from_commit_prepares_and_starts_separate_manual_workers - AssertionError: {\"outcome\": \"failure\", \"error\": \"worktree_outputs_on_mirror requires a symlink to [local path] [local path]"}\n2026-11-22T09:16:50.9074394Z   parent-session: storage preparation failed; fix the reported cause and retry the same --from[local path]  +  where 2 = CompletedProcess(args=['[local path]', '-c', '[local path]').returncode\n2026-11-22T09:16:50.9078487Z ===== 1 failed, 3433 passed, 1 skipped, 255 warnings in 715.15s (0:11:55) ======\n2026-11-22T09:16:52.3886622Z Selected: 3435; outcomes: `{\"failed\": 1, \"passed\": 3433, \"skipped\": 1}`.\n2026-11-22T09:16:52.3900683Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T09:16:52.4038570Z ##[error]Process completed with exit code 1.\n2026-11-22T09:16:53.0266575Z Selected: 3435; outcomes: `{\"failed\": 1, \"passed\": 3433, \"skipped\": 1}`.\n2026-11-22T09:16:53.0280630Z | Fixture | Executions | Setup s | Errors |\n2026-11-22T09:16:53.0443603Z ##[error]Process completed with exit code 2.\n2026-11-22T09:16:54.5901214Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45005616425,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45005616425/job/137162733780",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-22T09:27:58.5095338Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-22T09:27:58.5438340Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-22T09:27:58.5742556Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-22T09:27:58.5908976Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-22T09:27:58.6906788Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-22T09:28:01.8014049Z   digest-mismatch: error\n2026-11-22T09:28:02.4737303Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-22T09:28:17.6920078Z     \"incomplete/failed test observations\",\n2026-11-22T09:28:17.7042240Z     \"failed\": 1\n2026-11-22T09:28:17.7233734Z ##[error]Process completed with exit code 1.\n2026-11-22T09:28:17.7462409Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
