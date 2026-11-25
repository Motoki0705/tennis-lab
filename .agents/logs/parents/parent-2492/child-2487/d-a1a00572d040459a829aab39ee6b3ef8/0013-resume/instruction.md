## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204385483118668058_2026-11-25T13-48-36-881331941Z_1163401_i2487-ecdet-l-coco-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204385482836440906_2026-11-25T13-48-37-163559093Z_1163447_i2487-ecdet-x-coco-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204385482505987943_2026-11-25T13-48-37-494012056Z_1163508_i2487-ecdet-l-o365-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204385482195415864_2026-11-25T13-48-37-804584135Z_1163781_i2487-ecdet-x-o365-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204385481889489464_2026-11-25T13-48-38-110510535Z_1163961_i2487-ecpose-l-o365-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204385481606919777_2026-11-25T13-48-38-393080222Z_1164088_i2487-ecpose-x-o365-pt16-speed-threads4-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
