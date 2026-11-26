## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T073938Z_83daa37a1583_i2444-merge-cpu-r2",
      "name": "i2444-merge-cpu-r2",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 370.129,
      "log": "[local path]",
      "log_excerpt": "8.03s call     tests/unit/tennis_scene/generate_dataset/test_entrypoint.py::test_real_hydra_cli_fails_when_a_clip_fails"
    }
  ]
}
