[PR #2461](https://github.com/Motoki0705/tennis-lab/pull/2461) に正式レビューの修正2件をpushし、PR本文を更新しました。#2610 を含む `main` をマージ済みです。

ブランチ `codex/parent-1739-issue-2459-primary`  
HEAD `71a8a1fc4d2c181692f77175add37820dc94f62f`

[再生成した比較ページ](https://github.com/Motoki0705/tennis-lab/blob/71a8a1fc4d2c181692f77175add37820dc94f62f/outputs/pose-comparison-review-r1/index.html) で、未使用視点への再投影残差と分母を確認しました。

| Clip | 旧経路（中央値 / P95 / 分母） | 新経路（中央値 / P95 / 分母） |
| --- | --- | --- |
| `clip_052` | 7.5 px / 24.0 px / 1,076点 | 32.3 px / 142.1 px / 532点 |
| `clip_060` | 人物再構成が停止（0点） | 30.8 px / 165.9 px / 829点 |

`clip_052` の旧経路の集計は #2470 と一致しました。保存済みv3の `clip_012` は `event_detection` から採用中の `scene_assembly` をたどり、3,702点を集計できました。

関連pytestは94件成功（43.54秒、4件追加）。pre-commitとknowledge検査も通過しました。公開用16本の再生成とArtifact公開、#2469との合流は後続です。[新HEADのCIは実行中です。](https://github.com/Motoki0705/tennis-lab/actions/runs/45815004554)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2461,"head":"71a8a1fc4d2c181692f77175add37820dc94f62f","resume_on_success":false}}