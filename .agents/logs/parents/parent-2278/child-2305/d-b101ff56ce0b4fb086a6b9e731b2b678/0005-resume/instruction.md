## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204842394767435082_2026-11-20T06-53-25-232564917Z_2678642_i2305-gvhmr-build-and-measure-2-r4.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842394013169528_2026-11-20T06-53-25-986830471Z_2679362_i2305-gvhmr-build-and-measure-3a-r4.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842393308061770_2026-11-20T06-53-26-691938229Z_2679639_i2305-gvhmr-build-and-measure-3b-r4.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
