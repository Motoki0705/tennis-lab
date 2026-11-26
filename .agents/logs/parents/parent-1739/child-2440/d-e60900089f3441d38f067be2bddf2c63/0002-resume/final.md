[ドラフトPR #2450](https://github.com/Motoki0705/tennis-lab/pull/2450) を更新しました。ブランチは `codex/parent-1739-issue-2440-primary`、HEADは `74dbcca0b7dfd03ab28011d5da402aed13dfd5d4` です。CIは実行中です。

既定の診断形式を復元し、固定fixtureの期待値を変えずにCIの失敗を修正しました。全69クリップで既定診断276件と、4条件の選別マスク1,104件が保存結果に一致しました。関連79テスト、pre-commit、knowledge検査が通過しました。

疑似正解の69クリップ・1,205フレームで、**ダブルス幅・滞在25%**はbox再現率を89.34%から93.45%に、選手フレーム再現率を90.00%から94.13%に上げました。box適合率は93.99%から93.76%になり、非選手ラベルの選別が90件増えました。そのうち67件はアレー相当の位置でした。人による抜き取りの誤り率は未記録のため、採点は暫定です。[全条件・層別の結果と誤選別例](https://github.com/Motoki0705/tennis-lab/blob/74dbcca0b7dfd03ab28011d5da402aed13dfd5d4/knowledge/runs/run-i2440-player-selection/README.md)

既存経路のダブルス用には幅の拡張・25%を第一候補とします。020・058を含む3クリップが復帰し、新規停止は0件でした。短い断片への対処は、#2445で複数視点と時間方向の3D軌跡を使って検証する案です。採否は人の確認後に判断します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2450,"head":"74dbcca0b7dfd03ab28011d5da402aed13dfd5d4","resume_on_success":false}}