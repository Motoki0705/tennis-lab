## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204680327378668289_2026-11-22T03-54-32-621331710Z_130245_i2336-r2-vit-b32.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680327098825409_2026-11-22T03-54-32-901174590Z_130293_i2336-r2-vit-trt.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680326811229430_2026-11-22T03-54-33-188770569Z_130334_i2336-r2-hmr2-fp16.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680326514630579_2026-11-22T03-54-33-485369420Z_130382_i2336-r2-dino-grid.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680326215634010_2026-11-22T03-54-33-784365989Z_130426_i2336-r2-dino-plugin.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680325868991475_2026-11-22T03-54-34-131008524Z_130511_i2336-r2-hmr2-3a.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680325516929045_2026-11-22T03-54-34-483070954Z_130739_i2336-r2-hmr2-3b.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680325210466377_2026-11-22T03-54-34-789533622Z_130821_i2336-r2-vit-ncu-0.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680324869509851_2026-11-22T03-54-35-130490148Z_130897_i2336-r2-hmr2-ncu-0.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680324572173065_2026-11-22T03-54-35-427826934Z_130977_i2336-r2-dino-ncu-0.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680324266206890_2026-11-22T03-54-35-733793109Z_131063_i2336-r2-vit-ncu-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680323944887058_2026-11-22T03-54-36-055112941Z_131139_i2336-r2-hmr2-ncu-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680323635529962_2026-11-22T03-54-36-364470037Z_131216_i2336-r2-dino-ncu-1.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680323339668635_2026-11-22T03-54-36-660331364Z_131297_i2336-r2-vit-ncu-2.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680323028525681_2026-11-22T03-54-36-971474318Z_131371_i2336-r2-hmr2-ncu-2.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680322695779406_2026-11-22T03-54-37-304220593Z_131443_i2336-r2-hmr2-ncu-3a.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680322361653012_2026-11-22T03-54-37-638346987Z_131526_i2336-r2-hmr2-ncu-3b.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680322062664314_2026-11-22T03-54-37-937335685Z_131601_i2336-r2-dino-ncu-2.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    },
    {
      "job_id": "0-8204680321780621081_2026-11-22T03-54-38-219378918Z_131673_i2336-r2-dino-ncu-2-plugin.job",
      "state": "failed",
      "exit_code": 3,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": "[local path]"
    }
  ]
}
