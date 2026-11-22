## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204842680668697072_2026-11-20T06-48-39-331302927Z_2612555_i2303-vitpose-3a-r3.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842680381613157_2026-11-20T06-48-39-618386842Z_2612632_i2303-vitpose-3b-r3.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842680091142395_2026-11-20T06-48-39-908857604Z_2612701_i2303-hmr2-2-r3.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842679805963649_2026-11-20T06-48-40-194036350Z_2612771_i2303-hmr2-3a-r3.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842679522710353_2026-11-20T06-48-40-477289646Z_2612844_i2303-hmr2-3b-r3.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204842679229479399_2026-11-20T06-48-40-770520600Z_2612917_i2303-clip-3a-r3.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
