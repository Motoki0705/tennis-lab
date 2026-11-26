## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261125T175618Z_b6062ef3c7b9_ball-refiner-net-v1-generate-retry",
      "name": "ball-refiner-net-v1-generate-retry",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 163.713,
      "log": "[local path]",
      "log_excerpt": "{\"generated\": 1050, \"total\": 1280, \"seconds\": 133.4}\n{\"generated\": 1075, \"total\": 1280, \"seconds\": 135.3}\n{\"generated\": 1100, \"total\": 1280, \"seconds\": 138.9}\n{\"generated\": 1125, \"total\": 1280, \"seconds\": 140.9}\n{\"generated\": 1150, \"total\": 1280, \"seconds\": 143.6}\n{\"generated\": 1175, \"total\": 1280, \"seconds\": 147.5}\n{\"generated\": 1200, \"total\": 1280, \"seconds\": 150.1}\n{\"generated\": 1225, \"total\": 1280, \"seconds\": 154.0}\n{\"generated\": 1250, \"total\": 1280, \"seconds\": 155.6}\n{\"generated\": 1275, \"total\": 1280, \"seconds\": 159.5}\n{\"generated\": 1280, \"total\": 1280, \"seconds\": 159.8}\n{\"dataset\": \"[local path]", \"rallies\": 1280, \"frames\": 506468}"
    }
  ]
}
