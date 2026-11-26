## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T062127Z_83d0359d882e_scene-events-v3-config-audit",
      "name": "scene-events-v3-config-audit",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 96.425,
      "log": "[local path]",
      "log_excerpt": "\ntests/unit/utils/configuration/test_audit.py .                           [  8%]\ntests/unit/utils/configuration/test_contracts.py .                       [ 16%]\ntests/unit/utils/configuration/test_inventory.py ..........              [100%]\n\n- generated xml file: ./outputs/scene-events-v3/pytest-config-audit-555a246.xml -\n============================= slowest 5 durations ==============================\n24.88s call     tests/unit/utils/configuration/test_audit.py::test_repository_source_satisfies_current_audit\n0.27s call     tests/unit/utils/configuration/test_inventory.py::test_temporal_ball_evaluation_registers_the_shared_runtime_boundary\n\n(3 durations < 0.005s hidden.  Use -vv to show these durations.)\n======================== 12 passed in 92.47s (0:01:32) ========================="
    }
  ]
}
