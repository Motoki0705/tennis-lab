## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261125T175238Z_a0589922faa9_ball-refiner-net-v1-generate",
      "name": "ball-refiner-net-v1-generate",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 16.586,
      "log": "[local path]",
      "log_excerpt": "Error executing job with overrides: ['generation=net_contacts', 'paths.data_root=[local path]', 'generation.workers=8']\nconcurrent.futures.process._RemoteTraceback: \nTraceback (most recent call last):\n    raise ValueError(\nValueError: Position lies outside the canonical half-court cell grid: x=-9.219646453857422, y=-8.146438598632812, side='near'.\nTraceback (most recent call last):\nValueError: Position lies outside the canonical half-court cell grid: x=-9.219646453857422, y=-8.146438598632812, side='near'.\nSet the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace."
    }
  ]
}
