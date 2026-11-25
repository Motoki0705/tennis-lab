## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204389561547152306_2026-11-25T12-40-38-452847693Z_3832713_i2487-rfdetr-l-pt16-b1-matched-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204389561265977834_2026-11-25T12-40-38-734022165Z_3832782_i2487-rfdetr-l-pt16-b2-matched-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204389560981567683_2026-11-25T12-40-39-018432316Z_3832852_i2487-rfdetr-l-trt16-b1-matched-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204389560691772196_2026-11-25T12-40-39-308227803Z_3832920_i2487-rfdetr-l-trt16-b2-matched-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
