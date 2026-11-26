## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T113214Z_18ccbb4113fa_i2633-viewpoint-inputs",
      "name": "i2633-viewpoint-inputs",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 8.627,
      "log": "[local path]",
      "log_excerpt": "Traceback (most recent call last):\n    raise ValueError(\"The pool or original trajectory partition changed.\")\nValueError: The pool or original trajectory partition changed."
    }
  ]
}
