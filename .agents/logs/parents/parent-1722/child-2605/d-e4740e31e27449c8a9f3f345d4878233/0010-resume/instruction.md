## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T081104Z_a478ba7f9e17_scene-events-v3-review-major",
      "name": "scene-events-v3-review-major",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 2418.297,
      "log": "[local path]",
      "log_excerpt": "  },\n  \"clip_040_reference_changes\": {\n    \"artifact_id\": [\n      \"f75783a487049a6444b5cb3ed9c5084ed0c48d070a65ac06013c8857c4b56b1d\",\n      \"f6ab9b0a0caa90a95b97477f583ba1e918f4365c014f9e8badbe7b1d333b2124\"\n    ],\n    \"execution_key\": [\n      \"b6b956dcbbb11759dc1d527f5f271f4c5c299fc78d249b614c87ec79285f622c\",\n      \"8783a4d570dec901d64c626e3a3304b2001f564daaa3eaaae1a4d66c4440f3d9\"\n    ]\n  }\n}"
    }
  ]
}
