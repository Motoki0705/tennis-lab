## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204383228858026608_2026-11-25T14-26-11-141973391Z_1285078_i2487-ecdet-l-coco-trt16-fp32attn.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204383228564005184_2026-11-25T14-26-11-435994815Z_1285215_i2487-ecdet-x-coco-trt16-fp32attn.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204383228271215821_2026-11-25T14-26-11-728784178Z_1285356_i2487-ecdet-l-o365-trt16-fp32attn.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204383227982667381_2026-11-25T14-26-12-017332618Z_1285434_i2487-ecdet-x-o365-trt16-fp32attn.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
