## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204799080902372682_2026-11-20T18-55-19-097627317Z_2919771_i2335-r3-clip-fp8-timing.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799080648304534_2026-11-20T18-55-19-351695465Z_2919820_i2335-r3-gvhmr-fixed-timing-l145.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799080397544024_2026-11-20T18-55-19-602455975Z_2919857_i2335-r3-gvhmr-fixed-timing-l177.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799080146218012_2026-11-20T18-55-19-853781987Z_2919897_i2335-r3-clip-ncu-1-suffix.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799079893125015_2026-11-20T18-55-20-106874984Z_2919934_i2335-r3-clip-ncu-2-suffix.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799079630793376_2026-11-20T18-55-20-369206623Z_2920082_i2335-r3-clip-ncu-3b.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799079374224902_2026-11-20T18-55-20-625775097Z_2920221_i2335-r3-gvhmr-fixed-ncu-l145.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204799079111382642_2026-11-20T18-55-20-888617357Z_2920285_i2335-r3-gvhmr-fixed-ncu-l177.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
