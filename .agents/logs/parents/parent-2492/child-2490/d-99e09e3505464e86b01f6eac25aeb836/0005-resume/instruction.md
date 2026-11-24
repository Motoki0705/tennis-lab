## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T133254Z_8a496cc18184_sapiens2-onnx-fp32-rope",
      "name": "sapiens2-onnx-fp32-rope",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 850.64,
      "log": "[local path]",
      "log_excerpt": "sapiens2-1b export receipt {\"failure\": null, \"phase\": \"onnx_export\", \"wall_ms\": 138807.455434} cast targets {'FLOAT': 410, 'INT64': 966, 'FLOAT16': 80}\nsapiens2-5b export receipt {\"failure\": null, \"phase\": \"onnx_export\", \"wall_ms\": 686878.649834} cast targets {'FLOAT': 570, 'INT64': 1350, 'FLOAT16': 112}"
    }
  ]
}
