## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204380472820503651_2026-11-25T15-12-07-179496348Z_1537562_i2487-rtdetrv4-l-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204380472575366699_2026-11-25T15-12-07-424633300Z_1537589_i2487-rtdetrv4-x-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
