## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204420865584359985_2026-11-25T03-58-54-415640014Z_78461_person-speed-low-precision-detection.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204420865300709797_2026-11-25T03-58-54-699290202Z_78536_person-speed-low-precision-pose.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204420865009969488_2026-11-25T03-58-54-990030511Z_78605_person-speed-low-precision-body.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
