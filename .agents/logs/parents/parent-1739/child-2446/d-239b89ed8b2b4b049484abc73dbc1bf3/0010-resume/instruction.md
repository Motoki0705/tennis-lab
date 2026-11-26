## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T135952Z_72247c3aa886_i2446-review-r1-45-pidfd",
      "name": "i2446-review-r1-45-pidfd",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 71.0,
      "log": "[local path]",
      "log_excerpt": "757 nodes (708 run, 49 group) in ./knowledge/nodes — 0 error(s), 0 warning(s)."
    }
  ]
}
