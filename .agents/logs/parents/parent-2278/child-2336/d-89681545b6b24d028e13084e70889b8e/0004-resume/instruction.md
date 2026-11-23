## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204664054813652569_2026-11-22T08-25-45-186347430Z_2321832_i2336-r3-hmr2-ncu-3a-app.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204664054536945634_2026-11-22T08-25-45-463054365Z_2321888_i2336-r3-dino-ncu-2-app.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204664054236205593_2026-11-22T08-25-45-763794406Z_2321944_i2336-r3-dino-ncu-2-plugin-app.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
