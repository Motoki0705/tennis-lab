共有queueの稼働、GPUロック4ファイルの読み書き権限、固定計画のSHA-256を確認し、予約3 clip・9カメラ・38,421検出行のViTPose推論を投入しました。

完了結果を受け取り次第、固定設定で一度だけ採点し、全3 clipと校正補完clipを除く2 clipの結果を記録します。#2445向けの出力予定先は、このworktreeの `outputs/i2444/holdout/vitpose/stores/`、完了記録は同じ出力ルートの `complete.json` です。

PR #2449の完全HEADは `590c2377810762fdea0d5485d10307e612de51eb` です。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204503069350331066_2026-11-24T05-08-50-649668933Z_150553_i2444-holdout-vitpose.job"]}}