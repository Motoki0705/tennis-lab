## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T114610Z_7989777621e9_i2445-slot-review",
      "name": "i2445-slot-review",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 290.543,
      "log": "[local path]",
      "log_excerpt": "video_000/clip_057 285 / 285\nvideo_000/clip_058 765 / 765\nvideo_000/clip_059 471 / 475\nvideo_000/clip_060 313 / 313\nvideo_000/clip_061 487 / 487\nvideo_000/clip_062 273 / 430\nvideo_000/clip_063 341 / 341\nvideo_000/clip_064 1178 / 1252\nvideo_000/clip_065 355 / 355\nvideo_000/clip_066 613 / 613\nvideo_000/clip_067 543 / 543\nvideo_000/clip_068 462 / 462"
    }
  ]
}
