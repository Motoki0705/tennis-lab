## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204841827830284670_2026-11-20T07-02-52-169715329Z_2757180_i2304-dino-grid-2-v4.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204841827569138645_2026-11-20T07-02-52-430861354Z_2757247_i2304-dino-plugin-2-v4.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204841827243886683_2026-11-20T07-02-52-756113316Z_2757315_i2304-dino-export-grid-v4.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204841826981086319_2026-11-20T07-02-53-018913680Z_2757391_i2304-dino-grid-3a-v4.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204841826706861780_2026-11-20T07-02-53-293138219Z_2757458_i2304-dino-grid-3b-v4.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
