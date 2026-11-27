## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261127T002243Z_2113499c475e_i2664-full-config-retry",
      "name": "i2664-full-config-retry",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 393.494,
      "log": "[local path]",
      "log_excerpt": "FAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[blcs.scripts.generate_dataset-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[blcs.scripts.generate_dataset_samples-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[blcs.scripts.preview_augmentation-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[blcs.scripts.visualize-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[court_detection.scripts.train-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[player_detection.scripts.generate_dataset-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[player_detection.scripts.train-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[player_detection.scripts.evaluate-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_all_task_cli_output_contracts[player_detection.scripts.preview_dataset-defaults]\nFAILED tests/unit/utils/test_output_layout.py::test_visualize_mode_shares_log_identity_with_artifact[visualize-blcs]\nFAILED tests/unit/utils/test_output_layout.py::test_visualize_mode_shares_log_identity_with_artifact[predict-blcs]\n================== 21 failed, 736 passed in 293.86s (0:04:53) =================="
    }
  ]
}
