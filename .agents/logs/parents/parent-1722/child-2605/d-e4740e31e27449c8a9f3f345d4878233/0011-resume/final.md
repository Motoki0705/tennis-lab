`main` の `fcb97df3a64e4445efee3d943e419dd5bd928201` を取り込み、FastHMRのキャッシュ互換テストの競合を解消しました。関連59件とpre-commitのRuff・mypyが通過しました。統合HEAD `2999c0e743805fb2145ae6e03121f81b8c98df0c` を既存branch `codex/parent-1722-issue-2605-primary` にpushしました。 [PR #2610](https://github.com/Motoki0705/tennis-lab/pull/2610)

統合でコードのSHA-256が変わりました。Python CI通過後、統合HEADで69クリップを再生成し、現配布版と照合してから配布先とPRの実行証拠を更新します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2610,"head":"2999c0e743805fb2145ae6e03121f81b8c98df0c","resume_on_success":true,"next_step":"統合HEADでmultiview-v3の69クリップをCPUで再生成・再照合し、現配布版とイベント配列・coverageを比較してから配布先とPR #2610の実行証拠を更新する。"}}