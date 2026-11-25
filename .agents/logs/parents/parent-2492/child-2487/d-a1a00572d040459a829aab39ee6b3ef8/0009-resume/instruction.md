## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204406601172825111_2026-11-25T07-56-38-827174888Z_1736457_i2487-ecpose-l-o365-fixed-speed-13e508cb7.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204406600906716001_2026-11-25T07-56-39-093283998Z_1736525_i2487-ecpose-x-o365-fixed-speed-13e508cb7.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204406600618625364_2026-11-25T07-56-39-381374635Z_1736593_i2487-ecdet-l-coco-fixed-speed-13e508cb7.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204406600340455297_2026-11-25T07-56-39-659544702Z_1736660_i2487-ecdet-x-coco-fixed-speed-13e508cb7.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204406600020428136_2026-11-25T07-56-39-979571863Z_1736732_i2487-ecdet-l-o365-fixed-speed-13e508cb7.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204406599706080216_2026-11-25T07-56-40-293919783Z_1736800_i2487-ecdet-x-o365-fixed-speed-13e508cb7.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204405633049949023_2026-11-25T08-12-46-950050976Z_1848524_i2487-edgecrafter-trt16-raw-heads-diagnosis.job",
      "state": "failed",
      "exit_code": 1,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
