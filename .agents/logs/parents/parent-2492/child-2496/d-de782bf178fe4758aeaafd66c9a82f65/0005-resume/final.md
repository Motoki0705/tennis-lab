PR #2497のmajor 4件に対応し、`codex/parent-2492-issue-2496-primary` の `c7e43dd009993e6b53e8e6874c453d4ffb5bf642` にコミットしました。共通parity判定、バッチ終端のCUDA同期、3状態の公開記録、固定入力と計時順の照合を実装しました。固定入力manifestは公開済みのSHA-256と一致しています。 [PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

CPUテスト77件、新規15件が通過し、pre-commitとknowledge検査も成功しました。旧実測から13条件の計時ID順と、B=1の15条件のparity判定が一致することを確認しました。

現行モデル39条件の再計測を共有training queueに登録しました。終了後に新旧の差と実行可否をknowledgeとPR本文に反映し、pushしてCIに引き渡します。parityは現行閾値で投入し、Paretoの採否は親の判断待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204457787308838183_2026-11-24T17-43-32-691161816Z_3887924_person-speed-sync-detection.job","0-8204457787030960448_2026-11-24T17-43-32-969039551Z_3887991_person-speed-sync-pose.job","0-8204457786756298584_2026-11-24T17-43-33-243701415Z_3888064_person-speed-sync-body.job"]}}