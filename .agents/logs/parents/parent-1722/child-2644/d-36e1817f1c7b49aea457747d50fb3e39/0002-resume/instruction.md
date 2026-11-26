## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T115531Z_a05a161f7458_i2644-published-inputs",
      "name": "i2644-published-inputs",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 1136.411,
      "log": "[local path]",
      "log_excerpt": "Prepared video_000/clip_062 (430 frames)\nPrepared video_000/clip_063 (341 frames)\nPrepared video_000/clip_064 (1252 frames)\nPrepared video_000/clip_065 (355 frames)\nPrepared video_000/clip_066 (613 frames)\nPrepared video_000/clip_067 (543 frames)\nPrepared video_000/clip_068 (462 frames)\ne0c54222f9cfbfde278956e4438d158fe9fa4a17e92ec791b607bac396c24677  [local path] prepare elapsed 207.31 s\nAll 69 prepared arrays match #2600 by SHA-256, with matching event rows and splits.\nmanifest.json: OK\nSHA256SUMS: OK"
    }
  ]
}
