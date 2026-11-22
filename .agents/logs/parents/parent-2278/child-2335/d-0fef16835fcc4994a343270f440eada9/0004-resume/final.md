CLIP-ReIDの全25ケースとGVHMRのPyTorch全12ケースを実測し、速度・VRAM・nsys・単体精度を記録しました。代表バッチのncuは11件中9件でnsysとカーネル数が一致しました。 [実測表](https://github.com/Motoki0705/tennis-lab/blob/fb5b68adc8417b6d1281309285e0ec891b03b1ee/knowledge/nodes/tennis_scene/000079-run-i2335-trt-batch-sweep.md)・[metrics.json](https://github.com/Motoki0705/tennis-lab/blob/fb5b68adc8417b6d1281309285e0ec891b03b1ee/knowledge/runs/run-i2335-trt-batch-sweep/metrics.json)

CLIP-ReIDのFP8 B=16は中央値5.40 ms、P95 5.50 ms、2,958 crop/s、NVMLピーク増分297.1 MiBでした。ncuとnsysは115カーネルで一致し、IDは223/224件が一致しました。CLIP-ReIDの3つのTensorRTエンジンでB=1/16/32のprofileと層精度を照合し、B=0と33が終了値1で拒否されることを確認しました。 [検証記録](https://github.com/Motoki0705/tennis-lab/blob/fb5b68adc8417b6d1281309285e0ec891b03b1ee/knowledge/runs/run-i2335-trt-batch-sweep/validation.json)

GVHMRのTensorRT FP16はL=145と177を固定してもビルド中にSIGSEGVとなり、queueは終了値1を返しました。測定6ケース、ncu 2件、実機での範囲外拒否と層精度の確認が残っています。固定長・FP16・B=1/4/4のprofileを維持し、ビルダー最適化レベル0を指定した2ジョブを投入しました。各ジョブはビルドと計測が成功した場合にB=4のncuへ進みます。

`codex/parent-2278-issue-2335-primary` のコミットは `fb5b68adc8417b6d1281309285e0ec891b03b1ee`。CPUテスト89件、Ruff、knowledge検証が通り、raw出力から再集計したJSONはコミット内容と一致しました。PR作成は再試行の結果確認後です。自動再開は上限3回に達しているため、終端結果は同じsessionへの手動継続で引き継いでください。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204797946636145129_2026-11-20T19-14-13-363854870Z_3037453_i2335-r4-gvhmr-builder0-l145.job","0-8204797946383013931_2026-11-20T19-14-13-616986068Z_3037497_i2335-r4-gvhmr-builder0-l177.job"]}}