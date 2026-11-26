#2460 のCPU工程を全69 clipで完了しました。3D選手フレームは **133,887 → 147,105件**、停止は **14 → 11件** でした。clip_012・020・058で各4人が回復しました。全276視点の選別maskは #2440 と一致し、前段1,104件のartifact IDと #2458 のカメラ・ボール参照も照合しました。 [clip別の選手数・停止理由・処理時間](https://github.com/Motoki0705/tennis-lab/blob/bac14b71cfbbf0a9cd6bda9c5efef1fc97c84d61/knowledge/runs/run-meiji-doubles-player-reconstruction/cpu-clips.csv)

選別対象外に外見特徴の欠測があった6件は、検査条件を修正して再試行しました。全件の読戻し、pytest 391件、pre-commit、knowledge検査が通過しました。一時ストアでは参照の切替・復元とボール5配列のバイト一致を確認しました。

共有queueに **clip_000・020 のGPUパイロット**を登録しました。結果を受け取ったら、代表2件の公開と読戻しを検証し、全件の身体工程・参照切替・draft PRへ進みます。

branch `codex/parent-1739-issue-2460-primary`  
HEAD `bac14b71cfbbf0a9cd6bda9c5efef1fc97c84d61`

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204502006022507738_2026-11-24T05-26-33-977492261Z_204466_meiji-doubles-players-pilot.job"]}}