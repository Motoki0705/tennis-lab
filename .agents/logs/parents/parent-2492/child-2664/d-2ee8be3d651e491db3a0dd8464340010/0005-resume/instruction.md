## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261127T001637Z_7bc7ce85847b_i2664-full-config",
      "name": "i2664-full-config",
      "state": "failed",
      "exit_code": 1,
      "duration_seconds": 53.787,
      "log": "[local path]",
      "log_excerpt": "______ ERROR collecting tests/unit/utils/configuration/test_inventory.py _______\n    raise ValueError(\nE   ValueError: src.submodules.configuration.SubmoduleRuntimeConfig.rfdetr_l declares a Python runtime default; defaults must be composition-owned.\nERROR tests/unit/utils/configuration/test_audit.py - ValueError: src.submodul...\nERROR tests/unit/utils/configuration/test_contracts.py - ValueError: src.subm...\nERROR tests/unit/utils/configuration/test_discovery.py - ValueError: src.subm...\nERROR tests/unit/utils/configuration/test_inventory.py - ValueError: src.subm...\n!!!!!!!!!!!!!!!!!!! Interrupted: 4 errors during collection !!!!!!!!!!!!!!!!!!!!\n============================== 4 errors in 38.58s ==============================\nTraceback (most recent call last):\n    raise ValueError(\nValueError: src.submodules.configuration.SubmoduleRuntimeConfig.rfdetr_l declares a Python runtime default; defaults must be composition-owned."
    }
  ]
}
