## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261125T070233Z_387170703f99_fasthmr-render-review-6clips",
      "name": "fasthmr-render-review-6clips",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 2126.741,
      "log": "[local path]",
      "log_excerpt": "clip_057: 143 evaluation frames, 285 encoded source ticks, 4 videos\nclip_003: 185 evaluation frames, 369 encoded source ticks, 4 videos\nclip_013: 235 evaluation frames, 469 encoded source ticks, 4 videos\nclip_004: 294 evaluation frames, 586 encoded source ticks, 4 videos\nclip_040: 209 evaluation frames, 417 encoded source ticks, 4 videos\nclip_028: 645 evaluation frames, 1288 encoded source ticks, 4 videos"
    }
  ]
}
