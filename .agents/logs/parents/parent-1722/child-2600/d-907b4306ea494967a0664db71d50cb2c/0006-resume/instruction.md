## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T115528Z_4b14423a0aa5_meiji-v3-motion-final-coverage",
      "name": "meiji-v3-motion-final-coverage",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 507.285,
      "log": "[local path]",
      "log_excerpt": "Prepared video_000/clip_058 (765 frames)\nPrepared video_000/clip_059 (475 frames)\nPrepared video_000/clip_060 (313 frames)\nPrepared video_000/clip_061 (487 frames)\nPrepared video_000/clip_062 (430 frames)\nPrepared video_000/clip_063 (341 frames)\nPrepared video_000/clip_064 (1252 frames)\nPrepared video_000/clip_065 (355 frames)\nPrepared video_000/clip_066 (613 frames)\nPrepared video_000/clip_067 (543 frames)\nPrepared video_000/clip_068 (462 frames)\ne0c54222f9cfbfde278956e4438d158fe9fa4a17e92ec791b607bac396c24677  [local path]"
    }
  ]
}
