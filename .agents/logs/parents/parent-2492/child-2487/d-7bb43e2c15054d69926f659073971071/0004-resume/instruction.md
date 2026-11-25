## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204476497467538885_2026-11-24T12-31-42-532461114Z_1569088_i2487-rtdetrv4-l-pt32-accuracy.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476497096945226_2026-11-24T12-31-42-903054773Z_1569167_i2487-rtdetrv4-x-pt32-accuracy.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476496723166530_2026-11-24T12-31-43-276833469Z_1569247_i2487-rtdetrv4-l-pt16-accuracy.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476496327513662_2026-11-24T12-31-43-672486337Z_1569332_i2487-rtdetrv4-x-pt16-accuracy.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476495967023480_2026-11-24T12-31-44-032976519Z_1569406_i2487-rtdetrv4-l-trt16-build.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476495569056599_2026-11-24T12-31-44-430943400Z_1569480_i2487-rtdetrv4-x-trt16-build.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476495165376364_2026-11-24T12-31-44-834623635Z_1569563_i2487-rtdetrv4-l-trt16-accuracy.job",
      "state": "failed",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204476494760683743_2026-11-24T12-31-45-239316256Z_1569640_i2487-rtdetrv4-x-trt16-accuracy.job",
      "state": "failed",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
