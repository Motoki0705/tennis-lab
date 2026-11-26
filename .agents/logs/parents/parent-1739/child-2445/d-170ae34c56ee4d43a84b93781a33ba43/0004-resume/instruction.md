## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T082335Z_0310cd4124f1_i2445-holdout",
      "name": "i2445-holdout",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 78.878,
      "log": "[local path]",
      "log_excerpt": null
    }
  ]
}
