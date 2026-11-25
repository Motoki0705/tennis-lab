## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T110738Z_96f7ca821d94_multiview-player-evaluation",
      "name": "multiview-player-evaluation",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 306.161,
      "log": "[local path]",
      "log_excerpt": "e_px\": [771.7505493164062, 263.578125]}, \"right_eye\": {\"smpl_px\": [770.823360858868, 264.8091017886355], \"vitpose_px\": [769.5673828125, 263.71099853515625]}, \"left_ear\": {\"smpl_px\": [774.0053381361643, 264.86808784313405], \"vitpose_px\": [773.3214111328125, 263.6383972167969]}, \"right_ear\": {\"smpl_px\": [768.902849343293, 265.4656342347856], \"vitpose_px\": [767.3456420898438, 263.9156799316406]}}}}}}"
    }
  ]
}
