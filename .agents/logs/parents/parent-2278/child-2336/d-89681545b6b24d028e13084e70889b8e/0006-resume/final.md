#2371 の修正を含む `origin/main` を `codex/parent-2278-issue-2336-primary` に取り込み、`7aa9afeff42d59e86a7f74a11fa3ddbe896c0f62` をpushしました。PR本文の「Python CIの切り分け」も更新しました。 [PR #2373](https://github.com/Motoki0705/tennis-lab/pull/2373)

前回失敗した手動worker起動テストを含むストレージ関連223件が36.13秒、計測コードのCPUテスト86件が9.61秒で通過しました。対象テストのpre-commitと `git diff --check` も通過しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2373,"head":"7aa9afeff42d59e86a7f74a11fa3ddbe896c0f62","resume_on_success":true,"next_step":"対象HEADのPython CI成功を確認し、PR #2373のドラフトを解除して最終報告する。"}}