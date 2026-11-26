## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T134357Z_67fb419f89fa_i2446-review-r1-45-check",
      "name": "i2446-review-r1-45-check",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 641.722,
      "log": "[local path]",
      "log_excerpt": "757 nodes (708 run, 49 group) in ./knowledge/nodes — 0 error(s), 0 warning(s).\n=================================== FAILURES ===================================\nE       AssertionError: Traceback (most recent call last):\nE         AttributeError: module 'os' has no attribute 'pidfd_open'\nE         Traceback (most recent call last):\ntests/integration/tasks/annotation/test_scene_workflow.py:202: AssertionError\nFAILED tests/integration/tasks/annotation/test_scene_workflow.py::test_blank_cli_pins_policy_runs_isolated_condition_and_reports_without_publication\n============ 1 failed, 1875 passed, 1 warning in 612.24s (0:10:12) ============="
    }
  ]
}
