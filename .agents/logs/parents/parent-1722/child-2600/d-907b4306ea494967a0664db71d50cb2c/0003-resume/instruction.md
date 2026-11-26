## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T070346Z_236ec63d5ed9_meiji-v3-motion-inputs-r2",
      "name": "meiji-v3-motion-inputs-r2",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 318.877,
      "log": "[local path]",
      "log_excerpt": "Prepared video_000/clip_058 (765 frames)\nPrepared video_000/clip_059 (475 frames)\nPrepared video_000/clip_060 (313 frames)\nPrepared video_000/clip_061 (487 frames)\nPrepared video_000/clip_062 (430 frames)\nPrepared video_000/clip_063 (341 frames)\nPrepared video_000/clip_064 (1252 frames)\nPrepared video_000/clip_065 (355 frames)\nPrepared video_000/clip_066 (613 frames)\nPrepared video_000/clip_067 (543 frames)\nPrepared video_000/clip_068 (462 frames)\n2d4b16b9a2f38d7097759b763ae3b7c81018233168afffef06a2c06a7bdf2bd4  [local path]"
    }
  ]
}
