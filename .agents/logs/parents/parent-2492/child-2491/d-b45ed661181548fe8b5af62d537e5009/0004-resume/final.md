CameraHMRの提供資産3点を照合し、ロード時に検証するSHA-256を固定しました。実SMPLから6,890頂点とCOCO17をCPUで再構成し、平行移動が一度だけ反映されることを確認しました。対象テストとpre-commitは通過しました。 [Draft PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501#issuecomment-7132846129)

`codex/parent-2492-issue-2491-primary` の `ede444fd7a1b0b33d4624bc2be86109a1bcba4e2` をpushし、固定manifestの `clip_003` の8フレームでPT32・PT16の実推論確認をqueueに登録しました。

実推論の結果確認、TRT16の変換・parity試行、代表6 clipの比較動画が残っています。速度比較は #2496 の固定速度入力と基準値の公開待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204473201027913525_2026-11-24T13-26-38-972086474Z_2037423_i2491-fasthmr-pt32-pt16-smoke.job"]}}