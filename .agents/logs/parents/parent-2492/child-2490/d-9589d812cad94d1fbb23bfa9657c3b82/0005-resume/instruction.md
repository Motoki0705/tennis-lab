## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T235729Z_c92fccebb632_i2490-pmpose-rescore",
      "name": "i2490-pmpose-rescore",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 64.142,
      "log": "[local path]",
      "log_excerpt": "pmpose-l PT32 {'entire_score_report_unchanged': False, 'recall_unchanged': True, 'oks_pck_unchanged': True}\npmpose-l PT16 {'entire_score_report_unchanged': False, 'recall_unchanged': True, 'oks_pck_unchanged': True}\npmpose-l TRT16 {'entire_score_report_unchanged': False, 'recall_unchanged': True, 'oks_pck_unchanged': True}\npmpose-h PT32 {'entire_score_report_unchanged': False, 'recall_unchanged': True, 'oks_pck_unchanged': True}\npmpose-h PT16 {'entire_score_report_unchanged': False, 'recall_unchanged': True, 'oks_pck_unchanged': True}\npmpose-h TRT16 {'entire_score_report_unchanged': False, 'recall_unchanged': True, 'oks_pck_unchanged': True}"
    }
  ]
}
