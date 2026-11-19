[PR #2226](https://github.com/Motoki0705/tennis-lab/pull/2226) を更新しました。mainの#2225も取り込み済みで、Ready・merge可能です。

branch: `codex/person-feature-artifacts`  
HEAD: `121efe60d7035535789e09bbfb70f75e15971d3f`

5件とも対応しました。

1. 部分cacheでもDINO→ViTPose→CLIPの順序を維持。
2. load/import・追跡前にモデルと前処理identityを照合。
3. overrideのcrop設定をcache・完成markerへ反映。
4. CLIPの未使用pose依存を削除。
5. UI・Blender・benchmarkでpose v1/v2を許可。

CPU回帰1,694件、重点39件、CIで見つかった隔離不備を直したDINO試験28件、pre-commit成功。[CI全14ジョブ](https://github.com/Motoki0705/tennis-lab/actions/runs/44370023690)（Python含む）と[Knowledge CI全2ジョブ](https://github.com/Motoki0705/tennis-lab/actions/runs/44370023694)も成功しました。

GPU・再レビューは未実施、追加validatorは0/0/0、worktree cleanです。固定評価用raw reader2系の既存の旧版制約は未変更としてPR表へ記録しました。mergeは親へ返します。