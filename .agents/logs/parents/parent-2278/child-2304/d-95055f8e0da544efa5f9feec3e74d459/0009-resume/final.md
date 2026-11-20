INT8 v6は変換・精度検証・速度計測まで完了しました。中央値45.14 ms、P95 45.51 ms、22.14 fps、Peak VRAM 1.00 GiBで、Nsight Systemsは585カーネルを記録しました。FP32基準との照合ではIoU P05が0.9253、欠落11件・追加5件でした。Nsight Computeは45分でタイムアウトし、DRAM R/W、stall、Tensor Core activeは未採取です。[計測結果](https://github.com/Motoki0705/tennis-lab/blob/c80f8a4735f4db36b45fa9c0b2d7e5e342dd0f3d/knowledge/runs/run-i2304-dino-tensorrt/metrics.json) · [試行記録](https://github.com/Motoki0705/tennis-lab/blob/c80f8a4735f4db36b45fa9c0b2d7e5e342dd0f3d/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

CPUテスト98件とRuffが通過しました。変更は `codex/trt-dino` にステージ済みで、PRは未作成です。

`i2304-dino-grid-3a-ncu-v7` を投入しました。同じplanと固定入力でkernel replayを再試行し、採取できなければ診断用にapplication replayを実行します。完了後は585カーネルの一致とカウンタの取得状況を確認し、診断値は区別してknowledgeに記録します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204816680926065162_2026-11-20T14-01-59-073934837Z_1137002_i2304-dino-grid-3a-ncu-v7.job"]}}