## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T100319Z_0e179f3398ad_i2446-verify-and-render",
      "name": "i2446-verify-and-render",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 908.447,
      "log": "[local path]",
      "log_excerpt": "video_000/clip_020 217 1348557\nvideo_000/clip_023 579 4120371\nvideo_000/clip_028 430 2771966\nvideo_000/clip_036 135 912507\nvideo_000/clip_038 113 787651\nvideo_000/clip_039 291 1895950\nvideo_000/clip_049 374 2166192\nvideo_000/clip_058 255 1548235\nvideo_000/clip_060 105 668853\nvideo_000/clip_009 325 2170432\nvideo_000/clip_052 142 1033823\n./outputs/pose-comparison-bodies: 16 clips, 63,271,542 bytes"
    }
  ]
}
