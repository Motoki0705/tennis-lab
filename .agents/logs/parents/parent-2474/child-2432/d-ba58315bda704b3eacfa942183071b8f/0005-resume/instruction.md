## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T091015Z_7dceadbfa492_pr2478-ci-regressions",
      "name": "pr2478-ci-regressions",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 74.961,
      "log": "[local path]",
      "log_excerpt": "11.76s setup    tests/e2e/agents_ops/test_worker_followup.py::test_ci_resumes_only_for_failure_or_follow_on_work[success-True]\n5.26s call     tests/e2e/agents_ops/test_worker_followup.py::test_ci_resumes_only_for_failure_or_follow_on_work[success-True]"
    }
  ]
}
