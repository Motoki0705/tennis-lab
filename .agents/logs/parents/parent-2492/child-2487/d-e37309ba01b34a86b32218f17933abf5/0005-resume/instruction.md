## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204459650988321183_2026-11-24T17-12-29-011678816Z_3667397_i2487-rfdetr-l-pt32-b1-speed.job",
      "state": "cancelled",
      "exit_code": null,
      "log": null,
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204459650715102407_2026-11-24T17-12-29-284897592Z_3667465_i2487-rfdetr-l-pt32-b2-speed.job",
      "state": "cancelled",
      "exit_code": null,
      "log": null,
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204459650443192095_2026-11-24T17-12-29-556807904Z_3667534_i2487-rfdetr-l-pt16-b1-speed.job",
      "state": "cancelled",
      "exit_code": null,
      "log": null,
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204459650127155170_2026-11-24T17-12-29-872844829Z_3667601_i2487-rfdetr-l-pt16-b2-speed.job",
      "state": "cancelled",
      "exit_code": null,
      "log": null,
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204459649807334227_2026-11-24T17-12-30-192665772Z_3667674_i2487-rfdetr-l-trt16-b1-speed.job",
      "state": "cancelled",
      "exit_code": null,
      "log": null,
      "repro": null,
      "output_dir": null
    },
    {
      "job_id": "0-8204459649535148361_2026-11-24T17-12-30-464851638Z_3667747_i2487-rfdetr-l-trt16-b2-speed.job",
      "state": "cancelled",
      "exit_code": null,
      "log": null,
      "repro": null,
      "output_dir": null
    }
  ]
}
