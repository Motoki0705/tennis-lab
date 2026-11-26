## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T065934Z_25ce651a49fd_meiji-v3-motion-inputs",
      "name": "meiji-v3-motion-inputs",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 7.717,
      "log": "[local path]",
      "log_excerpt": "Traceback (most recent call last):\n    raise PathContractError(\nsrc.utils.configuration.errors.PathContractError: data_root must be a resolved absolute path; got PosixPath('./data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3')."
    }
  ]
}
