## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261125T175340Z_7daad550bc97_meiji-rule-ball-events",
      "name": "meiji-rule-ball-events",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 251.59,
      "log": "[local path]",
      "log_excerpt": "69/69 clips OK. Progress: [local path] peak_kib=1454896\n{\"method_version\": \"aac549c1181e8e34105a00d54fabe8f93f8ed700\", \"clips\": 69, \"valid_ball_frames\": 45958, \"events\": {\"shot\": 425, \"bounce\": 363, \"net\": 35}, \"flags\": {\"player_far\": 66}, \"progress_sha256\": \"65e5edab63d1f443dc2efab82878a52a2b67a7968486ee2cee6a578eabe5e7cb\"}"
    }
  ]
}
