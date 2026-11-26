## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T123234Z_03845a48192c_i2490-pmpose-parity-export",
      "name": "i2490-pmpose-parity-export",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 496.374,
      "log": "[local path]",
      "log_excerpt": "Failed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already `pip install kernels` and the kernels >= 0.11.1\nFailed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already `pip install kernels` and the kernels >= 0.11.1\nFailed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already `pip install kernels` and the kernels >= 0.11.1\nFailed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already `pip install kernels` and the kernels >= 0.11.1\nFailed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already `pip install kernels` and the kernels >= 0.11.1"
    }
  ]
}
