## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204539242105349297_2026-11-23T19-05-57-894650702Z_3918466_i2443-meiji4cam-clip.job",
      "state": "failed",
      "exit_code": null,
      "log": "[local path]",
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204539241715881427_2026-11-23T19-05-58-284118572Z_3918547_i2443-meiji4cam-solider.job",
      "state": "failed",
      "exit_code": null,
      "log": "[local path]",
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204539241340967491_2026-11-23T19-05-58-659032508Z_3918609_i2443-meiji4cam-kpr.job",
      "state": "failed",
      "exit_code": null,
      "log": "[local path]",
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204539240997373103_2026-11-23T19-05-59-002626896Z_3918685_i2443-meiji4cam-track.job",
      "state": "failed",
      "exit_code": null,
      "log": "[local path]",
      "repro": null,
      "output_dir": null
    }
  ]
}
