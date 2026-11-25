## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204388989148236245_2026-11-25T12-50-10-851763754Z_3908487_i2487-ecdet-l-o365-trt16-diagnosis-v2-requeued.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204388988887258277_2026-11-25T12-50-11-112741722Z_3908559_i2487-ecdet-l-coco-trt16-intermediate-probe-requeued.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
