## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T113758Z_1100cbe90df9_ball-paper-shape",
      "name": "ball-paper-shape",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 396.964,
      "log": "[local path]",
      "log_excerpt": null
    },
    {
      "job_id": "20261126T113758Z_b215433fb167_ball-paper-footprint",
      "name": "ball-paper-footprint",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 576.095,
      "log": "[local path]",
      "log_excerpt": "Decoded 60/276 camera-clips\nDecoded 80/276 camera-clips\nDecoded 100/276 camera-clips\nDecoded 120/276 camera-clips\nDecoded 140/276 camera-clips\nDecoded 160/276 camera-clips\nDecoded 180/276 camera-clips\nDecoded 200/276 camera-clips\nDecoded 220/276 camera-clips\nDecoded 240/276 camera-clips\nDecoded 260/276 camera-clips\n{\"frames\": 184912, \"video_count\": 276, \"audit_count\": 64, \"footprint_count\": 2208, \"opencv\": \"5.0.0\", \"seconds\": 575.1386614139774}"
    }
  ]
}
