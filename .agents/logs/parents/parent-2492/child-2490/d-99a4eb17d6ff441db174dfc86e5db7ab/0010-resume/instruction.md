## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204399428034176235_2026-11-25T09-56-11-965823764Z_2585778_i2490-rtmpose-m-low-precision-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399427688229043_2026-11-25T09-56-12-311770956Z_2585722_i2490-rtmpose-x-low-precision-speed.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399509493995139_2026-11-25T09-54-50-506004860Z_2573712_i2490-pmpose-l-pt16-speed-sync.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399509083352450_2026-11-25T09-54-50-916647549Z_2573813_i2490-pmpose-l-trt16-speed-sync.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399508674762212_2026-11-25T09-54-51-325237787Z_2573934_i2490-pmpose-h-pt16-speed-sync.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204399508312554524_2026-11-25T09-54-51-687445475Z_2574122_i2490-pmpose-h-trt16-speed-sync.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
