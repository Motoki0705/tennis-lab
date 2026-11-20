## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204808791920653876_2026-11-20T16-13-28-079346123Z_2041184_i2335-r2-clip-timing.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808791649586360_2026-11-20T16-13-28-350413639Z_2041225_i2335-r2-gvhmr-timing.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808791368097600_2026-11-20T16-13-28-631902399Z_2041270_i2335-r2-clip-ncu-0.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808791103005008_2026-11-20T16-13-28-896994991Z_2041307_i2335-r2-clip-ncu-1.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808790835095928_2026-11-20T16-13-29-164904071Z_2041344_i2335-r2-clip-ncu-2.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808790540958882_2026-11-20T16-13-29-459041117Z_2041419_i2335-r2-clip-ncu-3a.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808790171705597_2026-11-20T16-13-29-828294402Z_2041622_i2335-r2-clip-ncu-3b.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808789835131929_2026-11-20T16-13-30-164868070Z_2041708_i2335-r2-gvhmr-ncu-0-l145.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808789536315216_2026-11-20T16-13-30-463684783Z_2041781_i2335-r2-gvhmr-ncu-0-l177.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808789253728525_2026-11-20T16-13-30-746271474Z_2041855_i2335-r2-gvhmr-ncu-2-l145.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204808788926023142_2026-11-20T16-13-31-073976857Z_2041921_i2335-r2-gvhmr-ncu-2-l177.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
