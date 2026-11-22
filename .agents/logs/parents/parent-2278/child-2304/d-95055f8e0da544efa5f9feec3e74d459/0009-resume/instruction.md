## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204822146448313429_2026-11-20T12-30-53-551686570Z_638131_i2304-dino-grid-3a-v6.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}
