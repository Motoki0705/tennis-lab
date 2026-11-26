## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T062258Z_6948dbcef7e6_i2446-main-merge-check",
      "name": "i2446-main-merge-check",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 391.567,
      "log": "[local path]",
      "log_excerpt": "756 nodes (707 run, 49 group) in ./knowledge/nodes — 0 error(s), 0 warning(s)."
    }
  ]
}
