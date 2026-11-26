## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204383399993249512_2026-11-25T14-23-20-006750487Z_1262166_i2487-rtdetrv4-l-trt16-fp32attn-v1.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204383399731915466_2026-11-25T14-23-20-268084533Z_1262198_i2487-rtdetrv4-x-trt16-fp32attn-v1.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
