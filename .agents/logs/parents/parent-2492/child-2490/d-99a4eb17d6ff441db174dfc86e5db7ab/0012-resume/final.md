#2541を含むmainを取り込み、[PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) の既存branchへpushしました。新HEADは `a1a1fb742cbbc21b84f78661b570e5c5cf712287` です。`maintenance.lock` の競合修正を含め、経緯と旧CIログをPR本文の独立した節に記録しました。

前回失敗した論文フィルタのChromiumテスト1件、関連pytest 121件、pre-commitが通過しました。比較CSVとPareto図は再生成してバイト一致を確認しました。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45625528820) と [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/45625528493) は結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"a1a1fb742cbbc21b84f78661b570e5c5cf712287","resume_on_success":false}}