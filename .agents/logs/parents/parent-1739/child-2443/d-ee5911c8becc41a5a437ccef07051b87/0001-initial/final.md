[ドラフトPR #2451](https://github.com/Motoki0705/tennis-lab/pull/2451) を #2448 の上に作成しました。HEAD は `c8b959f3db1e2e87e3bb5ac37d944275004d0c43` です。

- run 8〜10の再現可能な7条件と保存済み現行経路を比較できるようにしました。旧BoT-SORT系4条件の除外理由、層別指標、#2446向けの3D ID射影採点を記録しました。 [比較記録](https://github.com/Motoki0705/tennis-lab/blob/c8b959f3db1e2e87e3bb5ac37d944275004d0c43/knowledge/runs/run-i2443-meiji-4cam-tracker-comparison/README.md)
- 現行経路の試行は3 clip・16 frame・486 box。raw/group IDF1は **0.8484／0.8527**、pair F1は **0.5333**、3D coverageは **34/64**、停止は **1/3 clip** でした。pytest 150件、pre-commit、knowledge整合検査が通り、試行CSVは再実行で一致しました。
- 全69 clipのCLIP・SOLIDER・KPR特徴生成と、それに続くCPU追跡を共有キューに登録しました。完走とCIは結果待ちです。#2442の全体公開後に1,205 frameを採点し、層別の人手誤り率と併せて採否基準案を更新します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204539242105349297_2026-11-23T19-05-57-894650702Z_3918466_i2443-meiji4cam-clip.job","0-8204539241715881427_2026-11-23T19-05-58-284118572Z_3918547_i2443-meiji4cam-solider.job","0-8204539241340967491_2026-11-23T19-05-58-659032508Z_3918609_i2443-meiji4cam-kpr.job","0-8204539240997373103_2026-11-23T19-05-59-002626896Z_3918685_i2443-meiji4cam-track.job"]}}