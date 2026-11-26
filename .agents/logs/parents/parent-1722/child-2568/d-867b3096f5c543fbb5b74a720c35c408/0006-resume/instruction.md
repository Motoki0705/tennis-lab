## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T010440Z_8bfcf27dbb18_meiji-net-contact-review",
      "name": "meiji-net-contact-review",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 219.78,
      "log": "[local path]",
      "log_excerpt": "scoring refiner3 shot tp=28 fp=0 fn=10 precision=1.0 recall=0.7368421052631579 f1=0.8484848484848485 mean_abs_error_ms=20.854166666666668\nscoring refiner3 bounce tp=32 fp=0 fn=13 precision=1.0 recall=0.7111111111111111 f1=0.8311688311688312 mean_abs_error_ms=21.896875\nscoring refiner3 net tp=0 fp=1 fn=4 precision=0.0 recall=0.0 f1=0.0 mean_abs_error_ms=None\nscoring rule shot tp=33 fp=5 fn=5 precision=0.868421052631579 recall=0.868421052631579 f1=0.868421052631579 mean_abs_error_ms=21.738888888888887\nscoring rule bounce tp=25 fp=0 fn=20 precision=1.0 recall=0.5555555555555556 f1=0.7142857142857143 mean_abs_error_ms=16.683333333333334\nscoring rule net tp=3 fp=1 fn=1 precision=0.75 recall=0.75 f1=0.75 mean_abs_error_ms=16.683333333333334\ntuning refiner3 shot tp=8 fp=0 fn=9 precision=1.0 recall=0.47058823529411764 f1=0.64 mean_abs_error_ms=4.170833333333333\ntuning refiner3 bounce tp=11 fp=0 fn=3 precision=1.0 recall=0.7857142857142857 f1=0.88 mean_abs_error_ms=10.616666666666667\ntuning refiner3 net tp=1 fp=0 fn=0 precision=1.0 recall=1.0 f1=1.0 mean_abs_error_ms=50.050000000000004\ntuning rule shot tp=14 fp=2 fn=3 precision=0.875 recall=0.8235294117647058 f1=0.8484848484848485 mean_abs_error_ms=10.725000000000001\ntuning rule bounce tp=12 fp=0 fn=2 precision=1.0 recall=0.8571428571428571 f1=0.9230769230769231 mean_abs_error_ms=8.341666666666667\ntuning rule net tp=0 fp=0 fn=1 precision=None recall=0.0 f1=0.0 mean_abs_error_ms=None"
    }
  ]
}
