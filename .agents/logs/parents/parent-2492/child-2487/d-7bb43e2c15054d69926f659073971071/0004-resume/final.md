RT-DETRv4-L/X の初回8件は、精度・速度の結果を残さず終了しました。PT32 は位置埋め込みがCPUに残って停止し、TensorRT は固定バッチ1のエンジンが共通ローダーの可変バッチ契約に合いませんでした。PT16・TRT16 の採点も前提成果物がなく停止しました。

両方を修正し、`codex/parent-2492-issue-2487-rtdetrv4` の `8434e5effaed52eb7934629f80683ad999a15331` にコミットしました。56件のテストと pre-commit が通過しました。#2496 の固定96画像はバッチ1・2で画像IDと順序が一致し、先頭2画像の L のCPU出力も両バッチで差分基準を満たしました。[修正・再現手順](https://github.com/Motoki0705/tennis-lab/blob/8434e5effaed52eb7934629f80683ad999a15331/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.md)

L/X の再計測を共有GPUキューに登録しました。#2495 の固定6,844画像で精度を、#2496 の固定96画像で速度・VRAMを計測します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204462013763719362_2026-11-24T16-33-06-236280637Z_3383926_i2487-rtdetrv4-l-suite-v2.job","0-8204462013456921802_2026-11-24T16-33-06-543078197Z_3384006_i2487-rtdetrv4-x-suite-v2.job"]}}