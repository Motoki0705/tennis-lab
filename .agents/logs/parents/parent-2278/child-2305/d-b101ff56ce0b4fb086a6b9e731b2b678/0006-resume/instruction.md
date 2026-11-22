## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204837885245029317_2026-11-20T08-08-34-754970682Z_3163088_i2305-gvhmr-build-and-measure-2-r5.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204837884971504911_2026-11-20T08-08-35-028495088Z_3163165_i2305-gvhmr-build-and-measure-3a-r5.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    },
    {
      "job_id": "0-8204837884702239052_2026-11-20T08-08-35-297760947Z_3163227_i2305-gvhmr-build-and-measure-3b-r5.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
