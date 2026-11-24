## parent CLI が確認した待機結果

対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2500,
  "head": "7f7a84ba619be8e06a972f38a491f0176bb18e7d",
  "status": "failure",
  "reasons": [
    "ci_python_failure"
  ],
  "failed_jobs": [
    {
      "name": "CPU tests (3)",
      "run_id": 45384629504,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45384629504/job/138412559578",
      "failed_steps": [
        "Test",
        "Summarize test measurements"
      ],
      "log_excerpt": "2026-11-24T11:49:00.9489444Z 5.57s call     tests/e2e/agents_ops/test_issue_supplement.py::test_hook_recovery_records_commands_and_keeps_failures_private[timeout]\n2026-11-24T11:49:01.4670150Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_current_fp32_cache_is_adopted_in_execute_and_load_modes - AssertionError: assert '32868efc800d...6008cd5b02351' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-24T11:49:01.4673872Z FAILED tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_batched_tensorrt_fp32_compatibility_requires_the_exact_sources - AssertionError: assert '32868efc800d...6008cd5b02351' == '6022ecaab6f2...a77e56c7e7539'\n2026-11-24T11:49:01.4676634Z FAILED tests/e2e/development/test_architecture_boundaries.py::test_repository_owned_forwards_are_computation_only - AssertionError: forward contract violations:\n2026-11-24T11:49:01.4682394Z ==== 3 failed, 3544 passed, 12 skipped, 298 warnings in 1064.43s (0:17:44) =====\n2026-11-24T11:49:04.2398995Z Selected: 3559; outcomes: `{\"failed\": 3, \"passed\": 3544, \"skipped\": 12}`.\n2026-11-24T11:49:04.2433727Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T11:49:04.2611413Z ##[error]Process completed with exit code 1.\n2026-11-24T11:49:05.2153806Z Selected: 3559; outcomes: `{\"failed\": 3, \"passed\": 3544, \"skipped\": 12}`.\n2026-11-24T11:49:05.2178048Z | Fixture | Executions | Setup s | Errors |\n2026-11-24T11:49:05.2363678Z ##[error]Process completed with exit code 2.\n2026-11-24T11:49:06.9818724Z   if-no-files-found: error",
      "log_unavailable_reason": null
    },
    {
      "name": "Python",
      "run_id": 45384629504,
      "attempt": 1,
      "url": "https://github.com/Motoki0705/tennis-lab/actions/runs/45384629504/job/138425931138",
      "failed_steps": [
        "Verify jobs and complete CPU coverage"
      ],
      "log_excerpt": "2026-11-24T11:55:29.8984940Z  * [new branch]          codex/ball3d-error-decomposition -> origin/codex/ball3d-error-decomposition\n2026-11-24T11:55:29.9050263Z  * [new branch]          codex/dispatcher-github-failure-investigation -> origin/codex/dispatcher-github-failure-investigation\n2026-11-24T11:55:29.9083276Z  * [new branch]          codex/main-ci-failure-loop  -> origin/codex/main-ci-failure-loop\n2026-11-24T11:55:29.9329712Z  * [new branch]          fix/issue-supplement-failures -> origin/fix/issue-supplement-failures\n2026-11-24T11:55:32.9057577Z   digest-mismatch: error\n2026-11-24T11:55:34.1372045Z   digest-mismatch: error\n    \"result\": \"failure\",\n2026-11-24T11:55:49.9010242Z     \"incomplete/failed test observations\",\n2026-11-24T11:55:49.9037720Z         \"failed\": 1\n2026-11-24T11:55:49.9077992Z     \"failed\": 3\n2026-11-24T11:55:49.9252607Z ##[error]Process completed with exit code 1.\n2026-11-24T11:55:49.9419218Z   if-no-files-found: error",
      "log_unavailable_reason": null
    }
  ],
  "missing_details": null
}

## 親からの補足: 土台の更新

待機明けの自動再開は、#2495 のPR #2498 と #2496 のPR #2497 の最新headがこのbranchに含まれていないため、親CLIに拒否された。親がこのbranchへ両方の最新head（#2498: b1011e406、#2497: d64962a28）をmergeした（このターンの開始時点のHEAD）。#2497 の d64962a28 は scene cache 互換を直す変更（`tests/integration/tennis_scene/test_fp32_cache_compatibility.py` 関連）を含む。

- 取り込み後のHEADで影響範囲のテストを通してから先へ進むこと。PRがあるならpushする。
- 土台はまだdraftで、今後も更新されうる。待機（`PARENT_RESULT` の waiting）で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終えること。
- 最終行の `PARENT_RESULT` は1行の正しいJSONにする（括弧の数を確認する）。
