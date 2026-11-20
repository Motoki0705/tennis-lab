## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204838163145463422_2026-11-20T08-03-56-854536577Z_3129954_i2303-vitpose-3a-r4.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204838162888051058_2026-11-20T08-03-57-111948941Z_3130036_i2303-hmr2-2-r4.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204838162611562645_2026-11-20T08-03-57-388437354Z_3130100_i2303-hmr2-3a-ncu-r4.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
