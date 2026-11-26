PMPose-L/HのPT16・TRT16速度計測を、#2497のマージ版harnessで共有queueに4件登録しました。各jobはbatch 1・4・8を計測します。今回のparityは「PT32未計測」と記録します。

`codex/parent-2492-issue-2490-pmpose` の `206aa346f555a60292a61b5284830f00e14a68cd` をpushし、リモートHEADとの一致を確認しました。[CIを起動済み](https://github.com/Motoki0705/tennis-lab/actions/runs/45589861670)。

固定入力384行とbatchごとの入力順を検証し、追加5件を含む64テストが12.18秒で通過しました。pre-commitでは対象外だったRuffとmypyを直接実行し、通過を確認しました。

GPU計測結果はqueue完了後に条件ごとの `report.json` で確認します。出力先は `outputs/pmpose/speed-pt16-trt16-sync` です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204399509493995139_2026-11-25T09-54-50-506004860Z_2573712_i2490-pmpose-l-pt16-speed-sync.job","0-8204399509083352450_2026-11-25T09-54-50-916647549Z_2573813_i2490-pmpose-l-trt16-speed-sync.job","0-8204399508674762212_2026-11-25T09-54-51-325237787Z_2573934_i2490-pmpose-h-pt16-speed-sync.job","0-8204399508312554524_2026-11-25T09-54-51-687445475Z_2574122_i2490-pmpose-h-trt16-speed-sync.job"]}}