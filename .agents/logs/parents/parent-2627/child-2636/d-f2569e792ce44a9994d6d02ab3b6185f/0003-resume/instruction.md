## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T114338Z_6191b7196f42_meiji-stopped-clip-replay",
      "name": "meiji-stopped-clip-replay",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 57.609,
      "log": "[local path]",
      "log_excerpt": "concurrent.futures.process._RemoteTraceback: \nTraceback (most recent call last):\n    raise DeviceSelectionError(\nsrc.utils.device.DeviceSelectionError: Explicit CUDA device 'cuda:0' was requested, but CUDA is unavailable.\nTraceback (most recent call last):\nsrc.utils.device.DeviceSelectionError: Explicit CUDA device 'cuda:0' was requested, but CUDA is unavailable."
    }
  ]
}
