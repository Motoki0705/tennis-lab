`origin/main` を取り込み、`person_unseen_freeze.py` の競合を解消しました。ボールイベント設定を省略できる読み込みと、ダブルス幅・滞在率の記録を両方残しました。保存済み比較の再現スクリプトも同じ読み込みに合わせました。[PR #2450](https://github.com/Motoki0705/tennis-lab/pull/2450)

2Dの足元選別と #2469 の3D軌跡選別が、ともに `|x| ≤ 5.485 m`、`|y| ≤ 16.885 m`、滞在25%を使うことを確認しました。CPUテスト118件が通過（36.13秒）し、pre-commitとknowledge検査も通過しました。比較値は保存済みの実測結果です。

ブランチ `codex/parent-1739-issue-2440-primary` の新HEAD `7b03cadf065fb46d413e4f036932afc8a245f0d4` をpushし、PR本文を更新しました。GitHubで競合解消を確認し、CIの結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2450,"head":"7b03cadf065fb46d413e4f036932afc8a245f0d4","resume_on_success":false}}