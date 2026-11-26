## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T114001Z_626be073a61f_i2634-player-selection",
      "name": "i2634-player-selection",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 3620.015,
      "log": "[local path]",
      "log_excerpt": "4cam/video_000/clip_045: replayed 4 conditions\n4cam/video_000/clip_017: replayed 4 conditions\n4cam/video_000/clip_038: replayed 4 conditions\n4cam/video_000/clip_065: replayed 4 conditions\n4cam/video_000/clip_063: replayed 4 conditions\n4cam/video_000/clip_060: replayed 4 conditions\n4cam/video_000/clip_005: replayed 4 conditions\n4cam/video_000/clip_031: replayed 4 conditions\n4cam/video_000/clip_037: replayed 4 conditions\n4cam/video_000/clip_057: replayed 4 conditions\n4cam/video_000/clip_051: replayed 4 conditions\n4cam/video_000/clip_027: replayed 4 conditions"
    }
  ]
}
