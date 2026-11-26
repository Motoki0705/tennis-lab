## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2571,
  "head": "4afdccc7c348a946cc5bb86aac583ae220ce652f",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (3)",
      "run_id": 45732828428,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45732828428/job/139619002073",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "camera_id\": \"far\", \"state\": \"ready\", \"reason\": null, \"current_attempt\": null, \"revision\": null, \"completion\": null, \"summary\": {\"total\": 4, \"processed\": 0, \"unprocessed\": 4, \"final_unknown\": 0, \"complete\": false}, \"processed_before\": 0, \"processed_after\": 0, \"new_processed\": [], \"worker_state\": \"ready\"}], \"failures\": [], \"reason\": \"annotating or continuing unprocessed camera frames\", \"stages\": []}\n2026-11-26T01:14:13.2967523Z   Traceback (most recent call last):\n2026-11-26T01:14:13.2976180Z       raise RuntimeError(f\"Local annotation needs recovery: {progress}\")\n57bd24cd0038b5c931b2a-1/launch.json', 'digest': '8c551f03e4f12972091ed1bef75a52cf49680831c8699491291dde088f1ad410', 'size_bytes': 1952}, 'exit': {'role': 'annotation', 'path': 'campaigns/local-fixture/jobs/e6511ff4bdfb9ecb35f0b367887eb95b33aa08724a357bd24cd0038b5c931b2a-1/exit.json', 'digest': '2a1ef5b86b00bd5c1905606ab261157864b647f0439b06dc4ab9e9012fcec90a', 'size_bytes': 904}, 'reason': None}]}\n2026-11-26T01:14:13.4397042Z ===== 1 failed, 2352 passed, 11 skipped, 303 warnings in 680.49s (0:11:20) =====\n2026-11-26T01:14:15.0679004Z Selected: 2364; outcomes: `{\"failed\": 1, \"passed\": 2352, \"skipped\": 11}`.\n2026-11-26T01:14:15.0704223Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T01:14:15.0889515Z ##[error]Process completed with exit code 1.\n2026-11-26T01:14:15.8730226Z Selected: 2364; outcomes: `{\"failed\": 1, \"passed\": 2352, \"skipped\": 11}`.\n2026-11-26T01:14:15.8756378Z | Fixture | Executions | Setup s | Errors |\n2026-11-26T01:14:15.8934920Z ##[error]Process completed with exit code 2.\n2026-11-26T01:14:17.4468616Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45732828428,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45732828428/job/139631380125",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-26T01:34:30.9837191Z  * [new branch]          codex/aggregate-preflight-path-errors -> origin/codex/aggregate-preflight-path-errors\n2026-11-26T01:34:30.9911904Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-26T01:34:30.9977974Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-26T01:34:31.0013637Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-26T01:34:31.0288607Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-26T01:34:33.9996249Z   digest-mismatch: error\n2026-11-26T01:34:34.7591227Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-26T01:34:49.5588680Z     \"incomplete/failed test observations\",\n2026-11-26T01:34:49.5692383Z     \"failed\": 1\n2026-11-26T01:34:49.5856324Z ##[error]Process completed with exit code 1.\n2026-11-26T01:34:49.6038224Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}
