#2759の打者推定を実装し、ready PRを作成しました。`Closes #2759` とCIの起動を確認済みです。 [PR #2769](https://github.com/Motoki0705/tennis-lab/pull/2769)

- Branch `codex/parent-2748-issue-2759-primary`
- HEAD `b11518eae36e640fa7573def42ee9c28d268edea`

22クリップの全111 shotを4視点で確認し、ルール実行前にGTを固定しました。valだけで閾値を選び、固定版でtestを評価しました。

| Split | 評価可能 | 正しく確定 | Coverage | 確定時正解率 | 誤確定率 |
|---|---:|---:|---:|---:|---:|
| val | 60 | 59 | 98.33% | 100% | 0/60 |
| test | 51 | 49 | 96.08% | 100% | 0/51 |

保留3件はすべて `too_far` で、接触窓内に打者本人の有効な手首・root観測がありませんでした。固定47/11/11 splitで元映像のフレーム区間の重複は0件です。3分割は同一試合に由来し、GTは単一作業者が確認しました。 [GT・監査・評価レポート](https://github.com/Motoki0705/tennis-lab/blob/b11518eae36e640fa7573def42ee9c28d268edea/knowledge/runs/run-i2759-v4-hitter-rule-eval/README.md)

PostgreSQL 17.12の試験用ingest runで、全69クリップの330件を保存・照会しました。再実行の更新は0件でした。ユニット83件、PostgreSQL結合9件、pre-commitのRuff・mypy、ナレッジ検証が成功し、専用Composeは停止済みです。検証コマンドはPR本文に記載しました。#2758の正式取り込み経路との通し確認は統合時に残ります。 [検証コマンド](https://github.com/Motoki0705/tennis-lab/pull/2769)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}