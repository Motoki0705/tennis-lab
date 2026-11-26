## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204399434903962229_2026-11-25T09-56-05-096037770Z_2584847_i2487-rfdetr-l-pt16-b1-sync-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399434489752884_2026-11-25T09-56-05-510247115Z_2584915_i2487-rfdetr-l-pt16-b2-sync-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399434067289058_2026-11-25T09-56-05-932710941Z_2584997_i2487-rfdetr-l-trt16-b1-sync-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399433646321135_2026-11-25T09-56-06-353678864Z_2585088_i2487-rfdetr-l-trt16-b2-sync-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
